""" Recommended Architecture
Separate RNN architecture is inspired by a popular RL repo
https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/POMDP/common/value_networks.py#L110
which has another branch to encode current state (and action)

Hidden state update functions get_hidden_state() is inspired by varibad encoder 
https://github.com/lmzintgraf/varibad/blob/master/models/encoder.py
"""

import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from pomdp_baselines.utils import helpers as utl
from pomdp_baselines.policies.rl import RL_ALGORITHMS
import pomdp_baselines.torchkit.pytorch_utils as ptu
from pomdp_baselines.policies.models.recurrent_critic import Critic_RNN
from pomdp_baselines.policies.models.recurrent_actor import Actor_RNN
from pomdp_baselines.utils import augmentation


class ModelFreeOffPolicy_Separate_RNN(nn.Module):
    """Recommended Architecture
    Recurrent Actor and Recurrent Critic with separate RNNs
    """

    ARCH = "memory"
    Markov_Actor = False
    Markov_Critic = False
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        encoder,
        algo_name,
        action_embedding_size,
        observ_embedding_size,
        reward_embedding_size,
        rnn_hidden_size,
        image_augmentation_type,
        image_augmentation_K,
        image_augmentation_M,
        image_augmentation_actor_critic_same_aug,
        dqn_layers,
        policy_layers,
        rnn_num_layers=1,
        lr=3e-4,
        gamma=0.99,
        tau=5e-3,
        # pixel obs
        image_encoder_fn=lambda: None,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.image_augmentation_type = image_augmentation_type
        self.image_augmentation_K = image_augmentation_K
        self.image_augmentation_M = image_augmentation_M
        self.image_augmentation_actor_critic_same_aug = image_augmentation_actor_critic_same_aug

        self.algo = RL_ALGORITHMS[algo_name](**kwargs.get(algo_name, {}), action_dim=action_dim)

        # Critics
        self.critic = Critic_RNN(
            obs_dim,
            action_dim,
            encoder,
            self.algo,
            action_embedding_size,
            observ_embedding_size,
            reward_embedding_size,
            rnn_hidden_size,
            dqn_layers,
            rnn_num_layers,
            image_encoder=image_encoder_fn(),  # separate weight
        )
        print("Critic: ")
        print(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        # target networks
        self.critic_target = deepcopy(self.critic)

        # Actor
        self.actor = Actor_RNN(
            obs_dim,
            action_dim,
            encoder,
            self.algo,
            action_embedding_size,
            observ_embedding_size,
            reward_embedding_size,
            rnn_hidden_size,
            policy_layers,
            rnn_num_layers,
            image_encoder=image_encoder_fn(),  # separate weight
        )        
        print("Actor: ")
        print(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        # target networks
        self.actor_target = deepcopy(self.actor)

    @torch.no_grad()
    def get_initial_info(self):
        return self.actor.get_initial_info()

    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        reward,
        obs,
        deterministic=False,
        return_log_prob=False,
    ):
        prev_action = prev_action.unsqueeze(0)  # (1, B, dim)
        reward = reward.unsqueeze(0)  # (1, B, 1)
        obs = obs.unsqueeze(0)  # (1, B, dim)

        current_action_tuple, current_internal_state = self.actor.act(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            reward=reward,
            obs=obs,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
        )

        return current_action_tuple, current_internal_state

    def forward(self, actions, rewards, observs, dones, masks):
        """
        For actions a, rewards r, observs o, dones d: (T+1, B, dim)
                where for each t in [0, T], take action a[t], then receive reward r[t], done d[t], and next obs o[t]
                the hidden state h[t](, c[t]) = RNN(h[t-1](, c[t-1]), a[t], r[t], o[t])
                specially, a[0]=r[0]=d[0]=h[0]=c[0]=0.0, o[0] is the initial obs

        The loss is still on the Q value Q(h[t], a[t]) with real actions taken, i.e. t in [1, T]
                based on Masks (T, B, 1)
        """
        assert (
            actions.dim()
            == rewards.dim()
            == dones.dim()
            == observs.dim()
            == masks.dim()
            == 3
        )
        assert (
            actions.shape[0]
            == rewards.shape[0]
            == dones.shape[0]
            == observs.shape[0]
            == masks.shape[0] + 1
        )
        num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss


        actor_normalize_pixel = self.actor.image_encoder.normalize_pixel
        critic_normalize_pixel = self.critic.image_encoder.normalize_pixel
        critic_target_normalize_pixel = self.critic_target.image_encoder.normalize_pixel

        """
        Critic_loss: 
            - Actor: observs_t_n
            - Critic_target: observs_t_n
            - Critic: observs_t
        Actor_loss:
            - observs_actor: observs_t of size 1

        K: Size of observs_t_n in Critic_loss
        M: Size of observs_t in Critic_loss -> observs_t size 1 in Actor_loss      
        """
        # Augment observations
        if self.image_augmentation_type != augmentation.AugmentationType.NONE:
            # set states for this method
            self.actor.image_encoder.normalize_pixel = False
            self.critic.image_encoder.normalize_pixel = False
            self.critic_target.image_encoder.normalize_pixel =  False
            
            # Calculate how many different augmentations are needed
            if self.image_augmentation_type == augmentation.AugmentationType.DIFFERENT_OVER_TIME:
                needed_augs = max(self.image_augmentation_K,self.image_augmentation_M)
            else:
                needed_augs = self.image_augmentation_K+self.image_augmentation_M
            if not self.image_augmentation_actor_critic_same_aug: # Add another aug if aug should be different for actor/critic
                needed_augs += 1;

            # Get augmented observations
            observs_new = augmentation.augment_observs(observs, self.image_augmentation_type, self.critic.image_encoder.shape, needed_augs)

            if self.image_augmentation_type == augmentation.AugmentationType.DIFFERENT_OVER_TIME:
                if self.image_augmentation_M >= self.image_augmentation_K:
                    observs_t = observs_new[:self.image_augmentation_M]
                    observs_t_n = observs_t[:self.image_augmentation_K]
                else:
                    observs_t_n = observs_new[:self.image_augmentation_K]
                    observs_t = observs_t_n[:self.image_augmentation_M]
            else:                
                observs_t_n = observs_new[:self.image_augmentation_K]
                observs_t = observs_new[self.image_augmentation_K:self.image_augmentation_K+self.image_augmentation_M]

            observs_actor = observs_t[0] if self.image_augmentation_actor_critic_same_aug else observs_new[-1] # important to use observs_t, not observs_t_n
        else:
            observs_t_n = observs
            observs_actor = observs
            observs_t = observs








        ### 1. Critic loss
        (q1_pred_list, q2_pred_list), q_target = self.algo.critic_loss(
            markov_actor=self.Markov_Actor,
            markov_critic=self.Markov_Critic,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observs_t=observs_t,
            observs_t_n=observs_t_n,
            actions=actions,
            rewards=rewards,
            dones=dones,
            gamma=self.gamma,
            image_augmentation_type = self.image_augmentation_type,
            image_augmentation_K = self.image_augmentation_K,
            image_augmentation_M = self.image_augmentation_M,
        )

        qf1_loss = 0
        qf2_loss = 0
        for q1_pred, q2_pred in zip(q1_pred_list, q2_pred_list):
            # masked Bellman error: masks (T,B,1) ignore the invalid error
            # this is not equal to masks * q1_pred, cuz the denominator in mean()
            # 	should depend on masks > 0.0, not a constant B*T
            q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
            q_target = q_target * masks
            # .TODO: Comparable to MSE? 
            qf1_loss += ((q1_pred - q_target) ** 2).sum() / num_valid  # TD error
            qf2_loss += ((q2_pred - q_target) ** 2).sum() / num_valid  # TD error

        self.critic_optimizer.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optimizer.step()

        ### 2. Actor loss
        policy_loss, log_probs = self.algo.actor_loss(
            markov_actor=self.Markov_Actor,
            markov_critic=self.Markov_Critic,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,            
            observs=observs_actor,
            actions=actions,
            rewards=rewards,
        )

        # recover states
        if self.image_augmentation_type != augmentation.AugmentationType.NONE:
            self.actor.image_encoder.normalize_pixel = actor_normalize_pixel
            self.critic.image_encoder.normalize_pixel = critic_normalize_pixel
            self.critic_target.image_encoder.normalize_pixel = critic_target_normalize_pixel


        # masked policy_loss
        policy_loss = (policy_loss * masks).sum() / num_valid

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        outputs = {
            "qf1_loss": qf1_loss.item(),
            "qf2_loss": qf2_loss.item(),
            "policy_loss": policy_loss.item(),
        }

        ### 3. soft update
        self.soft_target_update()

        ### 4. update others like alpha
        if log_probs is not None:
            # extract valid log_probs
            with torch.no_grad():
                current_log_probs = (log_probs[:-1] * masks).sum() / num_valid
                current_log_probs = current_log_probs.item()

            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.critic, self.critic_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.actor, self.actor_target, self.tau)

    def report_grad_norm(self):
        # may add qf1, policy, etc.
        return {
            "q_grad_norm": utl.get_grad_norm(self.critic),
            "q_rnn_grad_norm": utl.get_grad_norm(self.critic.rnn),
            "pi_grad_norm": utl.get_grad_norm(self.actor),
            "pi_rnn_grad_norm": utl.get_grad_norm(self.actor.rnn),
        }

    def update(self, batch):
        # all are 3D tensor (T,B,dim)
        actions, rewards, dones = batch["act"], batch["rew"], batch["term"]
        _, batch_size, _ = actions.shape
        if not self.algo.continuous_action:
            # for discrete action space, convert to one-hot vectors
            actions = F.one_hot(
                actions.squeeze(-1).long(), num_classes=self.action_dim
            ).float()  # (T, B, A)

        masks = batch["mask"]
        obs, next_obs = batch["obs"], batch["obs2"]  # (T, B, dim)

        # extend observs, actions, rewards, dones from len = T to len = T+1
        observs = torch.cat((obs[[0]], next_obs), dim=0)  # (T+1, B, dim)
        actions = torch.cat(
            (ptu.zeros((1, batch_size, self.action_dim)).float(), actions), dim=0
        )  # (T+1, B, dim)
        rewards = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), rewards), dim=0
        )  # (T+1, B, dim)
        dones = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), dones), dim=0
        )  # (T+1, B, dim)

        return self.forward(actions, rewards, observs, dones, masks)
