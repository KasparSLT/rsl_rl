#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from torch.serialization import add_safe_globals
from torch.distributions import MultivariateNormal
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization
from rsl_rl.utils import store_code_state


class InnerOnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu", history=None):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        actor_critic_class = eval(self.policy_cfg.pop("class_name")) 
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class( 
            num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        print("number of actions", self.env.num_actions)
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]  # this is relevant here
        # self.save_interval = self.cfg["save_interval"]  # this must go into the outer loop
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir  # this is the log dir from the outer loop, is that fine, can they just share the same log dir?
        print(f"Logging to {log_dir}")
        # writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0  # keep track of this also in the inner loop
        self.git_status_repos = [rsl_rl.__file__] # move to outer loop

        # initialize everything that is used in the step function and was former in the learn function 
        init_at_random_ep_len = False # dose it make sense to put this into the config file?
        if init_at_random_ep_len: 
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        
        self.obs, self.extras = self.env.get_observations()     # pool this with get_observations() above?
        self.critic_obs = self.extras["observations"].get("critic", self.obs)
        self.obs, self.critic_obs = self.obs.to(self.device), self.critic_obs.to(self.device)
        self.train_mode()  # Default is train mode, only when play, we switch to eval mode (in get_inference_policy())

        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.needed_observations = ["Mean_reward", "mean_episode_length", "mean_noise_std", "Loss/surrogate", "Loss/value_function", "Loss/learing_rate", "geometric observations (num geom joints)", "second geom joint"] # TODO
        self.history = history

        self.geom_map = env.get_geom_map()
        self.num_joints = env.get_geom_map().numel() 
        self.num_geom_joints = int(sum(self.geom_map))

        self.std_tensor = None


    def step(self, writer, current_it, num_steps: int = 1, init_at_random_ep_len: bool = False,):
        ep_infos = []

        # list of observations needed 

        # initialize feedback for the outer loop
        obs_outer = torch.zeros(len(self.needed_observations), dtype=torch.float, device=self.device) # TODO
        rewards_outer = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device) # reward for each env
        dones_outer = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device) # track terminated envs
        infos_outer = {} # TODO what exactly are they/is needed

        # Initialize observation tracking and step counts. E.g. stuff to check quality of geom actuators
        obs_series = torch.zeros((self.env.num_envs, 0, self.num_geom_joints), device=self.device) # TODO hardcoded for double pendulum
        step_counts = torch.zeros(self.env.num_envs, dtype=torch.int, device=self.device)
        weighted_std_sum = torch.zeros((self.env.num_envs,self.num_geom_joints), device=self.device)  # Sum of weighted stds for each env


        start_iter = current_it
        tot_iter = start_iter + num_steps
        for it in range(start_iter, tot_iter):
            start = time.time()
            # torch.inference_mode() should be already active from the outer loop, is there a way to check this?
            for i in range(self.num_steps_per_env):
                actions = self.alg.act(self.obs, self.critic_obs, self.std_tensor)
                self.obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                self.obs, self.critic_obs, rewards, dones = (
                    self.obs.to(self.device),
                    self.critic_obs.to(self.device),
                    rewards.to(self.device),
                    dones.to(self.device),
                )
                self.obs = self.obs_normalizer(self.obs)
                if "critic" in infos["observations"]:
                    self.critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                else:
                    self.critic_obs = self.obs
                self.alg.process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0
                
                # logging for the outer loop
                rewards_outer += rewards
                dones_outer += dones

                # Accumulate observations
                geom_obs = self.obs[:, self.num_joints:]
                geom_obs = geom_obs[:, self.geom_map.bool()]
                # print("geom_obs_dim", geom_obs.shape)
                obs_series = torch.cat([obs_series, geom_obs.unsqueeze(1)], dim=1)
                # print("obs_series_dim", obs_series.shape)
                step_counts += 1

                # Handle resets for done environments
                for steps in range(1, self.num_steps_per_env + 1):
                    # Identify environments that need to be reset and have exactly `steps` logged observations
                    reset_envs = (dones > 0) & (step_counts == steps)
                    if reset_envs.any():
                        reset_obs = obs_series[reset_envs]
                        # reset_counts = step_counts[reset_envs]

                        if reset_obs.shape[1] > 1:
                            # Calculate std and weight it by the number of steps
                            std_devs = reset_obs[:,-steps:].std(dim=1)
                            weighted_std_sum[reset_envs] += std_devs * steps

                        # Reset count done environments
                        step_counts[reset_envs] = 0


            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop
            self.alg.compute_returns(self.critic_obs)

            # with torch.enable_grad():   # enable gradient here for the update, will be disabled as soon as in outer loop again 
            mean_value_loss, mean_surrogate_loss = self.alg.update()

            obs_pos_log = self.obs[0, :self.num_joints]
            obs_pos_geom_joit_log = obs_pos_log[self.geom_map.bool()]


            learn_time = stop - start
            self.current_learning_iteration = it

            # calulate the final geom_std
            for steps in range(1, self.num_steps_per_env + 1):
                # Identify environments that need to be reset and have exactly `steps` logged observations
                relevant_envs = (dones > 0) & (step_counts == steps)
                if relevant_envs.any():
                    relevant_obs = obs_series[relevant_envs]
                    # relevant_counts = step_counts[relevant_envs]

                    if relevant_obs.shape[1] > 1:
                        # Calculate std and weight it by the number of steps
                        std_devs = relevant_obs[:,-steps:].std(dim=1)
                        weighted_std_sum[relevant_envs] += std_devs * steps
            
            # Calculate the mean of the std
            # print("weighted_std_sum", weighted_std_sum.shape)
            geom_std = torch.mean(weighted_std_sum, dim=0)/self.num_steps_per_env
            # print(f"Geom std: {geom_std}")

            
            
            if self.log_dir is not None:
                self.log(writer, locals())
            # if it % self.save_interval == 0:
            #     self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            self.logger_type = "wandb" # move to outer loop TODO:
            # if it == start_iter:
            if it == 0:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        writer.save_file(path)
            
            self.current_learning_iteration = it
        
        # modifiy the outer loop feedback
        rewards_outer /= (num_steps * self.num_steps_per_env)

        # create observations for the outer loop -> tensor of the in the list specified observations (the same order)
        # mean_reward = statistics.mean(rewards_outer)   # do I need to reset the buffer always in the beginning, is a reset done in the original
        if len(self.rewbuffer) > 0:
            mean_reward = statistics.mean(self.rewbuffer)
        else:
            mean_reward = 0
        if len(self.lenbuffer) > 0:
            mean_episode_length = statistics.mean(self.lenbuffer)
        else:
            mean_episode_length = 0

        # mean_episode_length = statistics.mean(self.lenbuffer)
        mean_std = self.alg.actor_critic.std.mean()
        # the two losses are defined above
        learning_rate = self.alg.learning_rate
        obs_outer = torch.tensor([mean_reward, mean_episode_length, mean_std, mean_surrogate_loss, mean_value_loss, learning_rate], device=self.device).unsqueeze(0).repeat(self.env.num_envs, 1)

        # positional observations
        obs_pos = self.obs[:, :self.num_joints]
        obs_pos_geom_joit = obs_pos[:, self.geom_map.bool()]

        obs_outer = torch.cat((obs_outer, obs_pos_geom_joit), dim=1)
        return obs_outer, rewards_outer, dones_outer, infos_outer



    def log(self, writer, locs: dict, width: int = 80, pad: int = 35):  # logging should be coordinated from outer loop
        # degenerate this function so it adds to an logger that is passed to it. Should be called from within the inner loop, but the logger should be passed from the outer loop
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]
        

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        
        if self.alg.actor_critic.std_tensor is None:
            mean_std = self.alg.actor_critic.std.mean()
        else:
            mean_std = torch.mean(self.alg.actor_critic.std_tensor)

        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        for i, obs in enumerate(locs["obs_pos_geom_joit_log"]):
            writer.add_scalar(f"Geometry/pos_geomjoint_{i}_obs", obs, locs["it"])
        
        for i, obs in enumerate(locs["geom_std"]):
            writer.add_scalar(f"Geometry/pos_geomjoint_{i}_std", obs, locs["it"])

    
        writer.add_scalar("Policy/Loss/value_function", locs["mean_value_loss"], locs["it"])
        writer.add_scalar("Policy/Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        writer.add_scalar("Policy/Loss/learning_rate", self.alg.learning_rate, locs["it"])
        writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        writer.add_scalar("Policy/Perf/total_fps", fps, locs["it"])
        writer.add_scalar("Policy/Perf/collection time", locs["collection_time"], locs["it"])
        writer.add_scalar("Policy/Perf/learning_time", locs["learn_time"], locs["it"])
        if len(self.rewbuffer) > 0:
            writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), locs["it"])
            writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                writer.add_scalar("Train/mean_reward/time", statistics.mean(self.rewbuffer), self.tot_time)
                writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs[self.lenbuffer]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(self.rewbuffer) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_steps'] - locs['it']):.1f}s\n""" # fromer num leaning itterations TODO
        )
        print(log_string)

    def save(self, path, infos=None): # move to outer loop, 
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos, # f"model_{self.current_learning_iteration}.pt"
            # "geometry_runner": self.geometry_runner.state_dict,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True): # make this an inner load function (take care of loading) that is called by the outer loop
        torch.serialization.add_safe_globals([MultivariateNormal])
        loaded_dict = torch.load(path, weights_only=True)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        
        # also make this optional
        # self.geometry_runner.load_state_dict(loaded_dict["geometry_runner"])
        return loaded_dict["infos"]
    
    def load_from_outer_loop(self, loaded_dict, load_optimizer=True):
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None): # is used in play.py to get the policy that is used. Call this from outher loop, after resetting the env to force the new geom on the env
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self): # I need simmelar in outer loop but not sure what to do with it here
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self): # I need simmelar in outer loop
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path): # find out what it is doing, provide interface in outer loop for train.py
        self.git_status_repos.append(repo_file_path)


