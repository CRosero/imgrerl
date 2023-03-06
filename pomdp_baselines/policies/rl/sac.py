import math
import torch
import torch.nn as nn
from torch.optim import Adam
from .base import RLAlgorithmBase
from pomdp_baselines.policies.models.actor import TanhGaussianPolicy
from pomdp_baselines.torchkit.networks import FlattenMlp
import torch.nn.functional as F
from pomdp_baselines.utils import helpers as utl
from pomdp_baselines.utils import augmentation

class SAC(RLAlgorithmBase):
    name = "sac"
    continuous_action = True
    use_target_actor = False

    def __init__(
        self,
        entropy_alpha=0.1,
        automatic_entropy_tuning=True,
        target_entropy=None,
        alpha_lr=3e-4,
        action_dim=None,
    ):

        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            if target_entropy is not None:
                self.target_entropy = float(target_entropy)
            else:
                self.target_entropy = -float(action_dim)
            init_alpha = torch.tensor(math.log(entropy_alpha))
            self.log_alpha_entropy = nn.Parameter(init_alpha)
            self.alpha_entropy_optim = Adam([self.log_alpha_entropy], lr=alpha_lr)
            self.alpha_entropy = self.log_alpha_entropy.exp().detach().item()
        else:
            self.alpha_entropy = entropy_alpha

    def update_others(self, current_log_probs):
        if self.automatic_entropy_tuning:
            alpha_entropy_loss = -self.log_alpha_entropy.exp() * (
                current_log_probs + self.target_entropy
            )

            self.alpha_entropy_optim.zero_grad()
            alpha_entropy_loss.backward()
            self.alpha_entropy_optim.step()
            self.alpha_entropy = self.log_alpha_entropy.exp().item()

        return {"policy_entropy": -current_log_probs, "alpha": self.alpha_entropy}

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        return TanhGaussianPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        if obs_dim is not None and action_dim is not None:
            input_size = obs_dim + action_dim
        qf1 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        qf2 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        return qf1, qf2

    def select_action(self, actor, observ, deterministic: bool, return_log_prob: bool):
        return actor(observ, False, deterministic, return_log_prob)

    @staticmethod
    def forward_actor(actor, observ):
        new_actions, _, _, log_probs = actor(observ, return_log_prob=True)
        return new_actions, log_probs  # (T+1, B, dim), (T+1, B, 1)





    def critic_loss(
        self,
        markov_actor: bool,
        markov_critic: bool,
        actor,
        actor_target,
        critic,
        critic_target,
        observs_t, # (T+1, B, C*H*W)
        observs_t_n,
        actions,
        rewards,
        dones,
        gamma,
        image_augmentation_type,
        image_augmentation_K,
        image_augmentation_M,
        next_observs=None,  # used in markov_critic
    ):
        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]

        # .TODO: Add noise to the deterministic action? Somewhere else?
        # .TODO: give unflattened to network directly?
        # .TODO: Does backward work if values are copied?
        # .TODO: torch.nograd used when augmenting -> correct?

        assert image_augmentation_type != augmentation.AugmentationType.NONE or (image_augmentation_K == 1 and image_augmentation_M==1)
        augmenting_image = image_augmentation_type != augmentation.AugmentationType.NONE





        with torch.no_grad():

            # .TODO: Refactor
            if augmenting_image:
                assert not markov_actor
                assert not markov_critic

                min_next_q_target_sum = 0

                for k in range(0,image_augmentation_K):
                    curr_new_actions, curr_new_log_probs = actor(
                        prev_actions=actions,
                        rewards=rewards,
                        observs=observs_t_n[k], 
                    )             
                    curr_next_q1, curr_next_q2 = critic_target( 
                        prev_actions=actions,
                        rewards=rewards,
                        observs=observs_t_n[k], 
                        current_actions=curr_new_actions,
                    )  # (T+1, B, 1)
                    curr_min_next_q_target = torch.min(curr_next_q1, curr_next_q2) + self.alpha_entropy * (-curr_new_log_probs)
                    min_next_q_target_sum += curr_min_next_q_target

                min_next_q_target = min_next_q_target_sum / image_augmentation_K # (T+1, B, 1)

                q_target = rewards + (1.0 - dones) * gamma * min_next_q_target  # next q
                q_target = q_target[1:]  # (T, B, 1)

            else:

                # get actions from actor, TODO: Add noise?
                # first next_actions from current policy,
                if markov_actor:
                    # .TODO: Not implemented
                    new_actions, new_log_probs = self.forward_actor(
                        actor, next_observs if markov_critic else observs
                    )
                else:
                    # (T+1, B, dim) including reaction to last obs
                    new_actions, new_log_probs = actor(
                        prev_actions=actions,
                        rewards=rewards,
                        observs=next_observs if markov_critic else observs_t_n, 
                    )
                
                # calculate q values of critic
                if markov_critic:  # (B, 1) 
                    # .TODO: Not implemented
                    next_q1 = critic_target[0](next_observs, new_actions)
                    next_q2 = critic_target[1](next_observs, new_actions)
                else:
                    next_q1, next_q2 = critic_target( 
                        prev_actions=actions,
                        rewards=rewards,
                        observs=observs_t_n, 
                        current_actions=new_actions,
                    )  # (T+1, B, 1)    

                min_next_q_target = torch.min(next_q1, next_q2)
                min_next_q_target += self.alpha_entropy * (-new_log_probs)  # (T+1, B, 1)


                # q_target: (T, B, 1)
                q_target = rewards + (1.0 - dones) * gamma * min_next_q_target  # next q
                if not markov_critic:
                    q_target = q_target[1:]  # (T, B, 1)










        if markov_critic:
            # .TODO: Not implemented
            q1_pred = critic[0](observs, actions)
            q2_pred = critic[1](observs, actions)
        else:
            if augmenting_image:
                q1_pred_list = []
                q2_pred_list = []
                for m in range(0, image_augmentation_M):
                    # Q(h(t), a(t)) (T, B, 1)
                    curr_q1_pred, curr_q2_pred = critic(
                        prev_actions=actions,
                        rewards=rewards,
                        observs=observs_t[m],
                        current_actions=actions[1:],
                    )  # (T, B, 1)
                    q1_pred_list.append(curr_q1_pred)
                    q2_pred_list.append(curr_q2_pred)
            else: 
                # Q(h(t), a(t)) (T, B, 1)
                q1_pred_list, q2_pred_list = critic(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs_t,
                    current_actions=actions[1:],
                )  # (T, B, 1)
                q1_pred_list = [q1_pred_list]
                q2_pred_list = [q2_pred_list]


        return (q1_pred_list, q2_pred_list), q_target



    def actor_loss(
        self,
        markov_actor: bool,
        markov_critic: bool,
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions=None,
        rewards=None,
    ):

        if markov_actor:
            new_actions, log_probs = self.forward_actor(actor, observs)
        else:
            new_actions, log_probs = actor(
                prev_actions=actions, rewards=rewards, observs=observs
            )  # (T+1, B, A)

        if markov_critic:
            q1 = critic[0](observs, new_actions)
            q2 = critic[1](observs, new_actions)
        else:
            q1, q2 = critic(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=new_actions,
            )  # (T+1, B, 1)

        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,1)

        policy_loss = -min_q_new_actions
        policy_loss += self.alpha_entropy * log_probs

        if not markov_critic:
            policy_loss = policy_loss[:-1]  # (T,B,1) remove the last obs

        
        return policy_loss, log_probs 

    #### Below are used in shared RNN setting
    def forward_actor_in_target(self, actor, actor_target, next_observ):
        return self.forward_actor(actor, next_observ)

    def entropy_bonus(self, log_probs):
        return self.alpha_entropy * (-log_probs)



