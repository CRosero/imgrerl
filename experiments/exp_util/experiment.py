import numpy as np
import torch
import time
import mbrl_envs

import pomdp_baselines.torchkit.pytorch_utils as ptu
from pomdp_baselines.policies.models.policy_rnn import ModelFreeOffPolicy_Separate_RNN as Policy_RNN
from pomdp_baselines.policies.models.policy_mlp import ModelFreeOffPolicy_MLP as Policy_MLP
from pomdp_baselines.policies.models import AGENT_ARCHS
from pomdp_baselines.buffers.seq_replay_buffer_vanilla import SeqReplayBuffer
from pomdp_baselines.buffers.seq_replay_buffer_efficient import RAMEfficient_SeqReplayBuffer
from pomdp_baselines.torchkit.networks import ImageEncoder
from pomdp_baselines.utils import helpers as utl

from experiments.exp_util.config_dict import ConfigDict

from torch.nn import functional as F

from torchsummary import summary

import torchvision.transforms as T
from PIL import Image
from datetime import datetime


class Experiment:

    @staticmethod
    def get_default_config():
        config = ConfigDict()

        config.seed = 0
        config.cuda_id = 0

        config.add_subconf("env", ConfigDict())
        config.env.env = "cheetah-run"
        config.env.obs_type = "image"

        config.add_subconf("agent", ConfigDict())
        config.agent.algo_name = "sac"
        config.agent.encoder = "lstm"
        config.agent.action_embedding_size = 32
        config.agent.observ_embedding_size = 128
        config.agent.reward_embedding_size = 32
        config.agent.rnn_hidden_size = 512
        config.agent.dqn_layers = [512, 512, 512]
        config.agent.policy_layers = [512, 512, 512]
        config.agent.lr = 0.0003
        config.agent.gamma = 0.99
        config.agent.tau = 0.005

        config.agent.entropy_alpha = 1.0

        config.add_subconf("rl", ConfigDict())
        config.rl.sampled_seq_len = 64
        config.rl.buffer_size = 1e6
        config.rl.batch_size = 32
        config.rl.rollouts_per_iter = 1
        config.rl.num_updates_per_iter = 100
        config.rl.num_init_rollouts = 5

        config.add_subconf("eval", ConfigDict())
        config.eval.interval = 20
        config.eval.num_rollouts = 20

        config.finalize_adding()
        return config

    def __init__(self,
                 config):

        cuda_id = config.cuda_id  # -1 if using cpu
        ptu.set_gpu_mode(torch.cuda.is_available() and cuda_id >= 0, cuda_id)

        domain_name, task_name = config.env.env.split("-")
        if config.env.obs_type == "state":
            obs_type = mbrl_envs.ObsTypes.STATE
        elif config.env.obs_type == "position":
            obs_type = mbrl_envs.ObsTypes.POSITION
        elif config.env.obs_type == "image":
            obs_type = mbrl_envs.ObsTypes.IMAGE
        else:
            raise ValueError("Unknown obs_type: {}".format(config.env.obs_type))
        self.env = mbrl_envs.make(domain_name=domain_name,
                                  task_name=task_name,
                                  seed=config.seed,
                                  action_repeat=-1,  # env default
                                  obs_type=obs_type,
                                  no_lists=True,
                                  old_gym_return_type=False)
        self._action_repeat = self.env.action_repeat
        self._max_trajectory_len = self.env.max_seq_length
        act_dim = self.env.action_space.shape[0]
        
        # TODO: Magic Number
        if config.env.obs_type == "image":
            #obs_dim = self.env.observation_space.shape
            obs_dim = self.env.observation_space.shape[0]*self.env.observation_space.shape[1]*self.env.observation_space.shape[2]
        else: 
            obs_dim = self.env.observation_space.shape[0]

        
        
        
        if config.agent.encoder == "mlp":
            self.agent = Policy_MLP(obs_dim=obs_dim,
                                    action_dim=act_dim,
                                    algo_name=config.agent.algo_name,
                                    action_embedding_size=config.agent.action_embedding_size,
                                    observ_embedding_size=config.agent.observ_embedding_size,
                                    reward_embedding_size=config.agent.reward_embedding_size,
                                    rnn_hidden_size=config.agent.rnn_hidden_size,
                                    dqn_layers=config.agent.dqn_layers,
                                    policy_layers=config.agent.policy_layers,
                                    lr=config.agent.lr,
                                    gamma=config.agent.gamma,
                                    tau=config.agent.tau,
                                    sac={"entropy_alpha": config.agent.entropy_alpha}).to(ptu.device)
        else:
            #cnn_no_fc.input_size
            #summary(model=cnn_no_fc, input_size = (3, 32, 32))

            # TODO: Do not use resnet; Magic number?
            # add image encoder when taking image as input
            image_enc = lambda: None
            if obs_type == mbrl_envs.ObsTypes.IMAGE:
                image_enc = lambda: ImageEncoder(
                    self.env.observation_space.shape,
                    embed_size=100, # Ouptut length after linear layer
                    depths=[8, 16], # Channels of Layers?
                    kernel_size=2,
                    stride=1,
                    activation="relu",
                    from_flattened=True,
                    normalize_pixel=True,
                )
                """
                model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
                cnn_no_fc = torch.nn.Sequential(*(list(model.children())[:-1]))
                cnn_no_fc.embed_size = 2048
                image_enc = lambda: cnn_no_fc
                #image_enc = cnn_no_fc
                """


            self.agent = Policy_RNN(obs_dim=obs_dim,
                                    action_dim=act_dim,
                                    encoder=config.agent.encoder,
                                    algo_name=config.agent.algo_name,
                                    action_embedding_size=config.agent.action_embedding_size,
                                    observ_embedding_size=config.agent.observ_embedding_size,
                                    reward_embedding_size=config.agent.reward_embedding_size,
                                    rnn_hidden_size=config.agent.rnn_hidden_size,
                                    dqn_layers=config.agent.dqn_layers,
                                    policy_layers=config.agent.policy_layers,
                                    lr=config.agent.lr,
                                    gamma=config.agent.gamma,
                                    tau=config.agent.tau,
                                    image_encoder_fn=image_enc,
                                    sac={"entropy_alpha": config.agent.entropy_alpha}).to(ptu.device)

        self._num_updates_per_iter = config.rl.num_updates_per_iter
        sampled_seq_len = config.rl.sampled_seq_len
        buffer_size = config.rl.buffer_size
        self._batch_size = config.rl.batch_size

        num_init_rollouts_pool = config.rl.num_init_rollouts
        self._num_rollouts_per_iter = config.rl.rollouts_per_iter
        self._n_env_steps_total = 0

        self.replay_buffer = RAMEfficient_SeqReplayBuffer(max_replay_buffer_size=int(buffer_size),
                                             observation_dim=obs_dim,
                                             action_dim=act_dim,
                                             sampled_seq_len=sampled_seq_len,
                                             sample_weight_baseline=0.0,
                                             observation_type=np.uint8) # TODO: Adaptable?

        avg_rewards, env_steps = self._collect_rollouts(
            num_rollouts=num_init_rollouts_pool, random_actions=True, train_mode=True
        )
        self._n_env_steps_total += (env_steps * num_init_rollouts_pool)

        # evaluation parameters
        self._last_eval_num_iters = 0
        self._log_interval = config.eval.interval
        self._eval_num_rollouts = config.eval.num_rollouts

    @torch.no_grad()
    def _collect_rollouts(self, num_rollouts, random_actions=False, deterministic=False, train_mode=True):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param
            random_actions: whether to use policy to sample actions, or randomly sample action space
            deterministic: deterministic action selection?
            train_mode: whether to train (stored to buffer) or test
        """
        if not train_mode:
            assert random_actions == False and deterministic == True

        total_steps = 0
        total_rewards = 0.0

        for idx in range(num_rollouts):
            steps = 0
            rewards = 0.0
            obs, info = self.env.reset()
            obs = ptu.from_numpy(obs) 
            obs = obs.reshape(1, obs.shape[-1])
            done_rollout = False

            # get hidden state at timestep=0, None for mlp

            if self.agent.ARCH == AGENT_ARCHS.Memory:
                # get hidden state at timestep=0, None for markov
                # NOTE: assume initial reward = 0.0 (no need to clip)
                action, reward, internal_state = self.agent.get_initial_info()

            if train_mode:
                # temporary storage
                obs_list, act_list, rew_list, next_obs_list, term_list = ([], [], [], [], [])

            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor(
                        [self.env.action_space.sample()]
                    )  # (1, A) for continuous action, (1) for discrete action
                else:
                    # policy takes hidden state as input for memory-based actor,
                    # while takes obs for markov actor
                    if self.agent.ARCH == AGENT_ARCHS.Memory:
                        # print("\n mem-based actor exp obs =", obs)
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward,
                            obs=obs,
                            deterministic=False,
                        )
                    else:
                        # print("\n markov exp obs =", obs)
                        action, _, _, _ = self.agent.act(obs, deterministic=False)








                # observe reward and next obs (B=1, dim)
                next_obs, reward, done, info = utl.env_step(self.env, action.squeeze(dim=0))
                
                # NOTE: Weird images
                #transform = T.ToPILImage() # [1, 3, 64, 64]
                #img = transform(next_obs.view((3, 64, 64)))
                #img.save(f"images/image_test_{datetime.now()}.png")

                # print("exp next_obs =", next_obs)
                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True





                # update statistics
                steps += 1
                rewards += reward.item()

                # early stopping env: such as rmdp, pomdp, generalize tasks. term ignores timeout
                term = (
                    False
                    if "TimeLimit.truncated" in info or steps >= self._max_trajectory_len
                    else done_rollout
                )

                if train_mode:
                    # append tensors to temporary storage
                    obs_list.append(obs)  # (1, dim)
                    act_list.append(action)  # (1, dim)
                    rew_list.append(reward)  # (1, dim)
                    term_list.append(term)  # bool
                    next_obs_list.append(next_obs)  # (1, dim)

                # set: obs <- next_obs
                obs = next_obs.clone()

            if train_mode:
                # add collected sequence to buffer
                self.replay_buffer.add_episode(
                    observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                    actions=ptu.get_numpy(torch.cat(act_list, dim=0)),  # (L, dim)
                    rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                    terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                    next_observations=ptu.get_numpy(
                        torch.cat(next_obs_list, dim=0)
                    ),  # (L, dim)
                )
            total_steps += steps
            total_rewards += rewards

        total_steps *= self._action_repeat
        return total_rewards / num_rollouts, total_steps / num_rollouts

    def _update(self, num_updates):
        rl_losses_agg = {}
        t0 = time.time()
        for update in range(num_updates):
            # sample random RL batch: in transitions
            batch = ptu.np_to_pytorch_batch(self.replay_buffer.random_episodes(self._batch_size))
            # RL update
            rl_losses = self.agent.update(batch)

            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)
        # statistics
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        rl_losses_agg["time"] = time.time() - t0
        return rl_losses_agg

    def iterate(self, iteration):

        log_dict = {}

        avg_returns, avg_steps = self._collect_rollouts(num_rollouts=self._num_rollouts_per_iter, train_mode=True)
        self._n_env_steps_total += (avg_steps * self._num_rollouts_per_iter)
        log_dict["collect/return"] = avg_returns
        log_dict["collect/steps"] = avg_steps
        log_dict["collect/total_steps"] = self._n_env_steps_total


        print("Collecting train stats")
        train_stats = self._update(int(self._num_updates_per_iter))
        log_dict.update({f"train/{k}": v for k, v in train_stats.items()})
        if iteration % self._log_interval == 0:
            avg_returns, avg_steps = self._collect_rollouts(
                num_rollouts=self._eval_num_rollouts,
                train_mode=False,
                random_actions=False,
                deterministic=True,
            )
            log_dict["eval/return"] = avg_returns
            log_dict["eval/steps"] = avg_steps

        return log_dict
