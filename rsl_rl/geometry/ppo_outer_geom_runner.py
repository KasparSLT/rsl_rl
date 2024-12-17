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


import torch
from torch.distributions import MultivariateNormal

from rsl_rl.runners import InnerOnPolicyRunner



class GeometryRunnerPPO:
    '''eins nicer text für die formalität'''
    
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.scale_mean = 0.1
        self.scale_cholseky = 0.1

        self.history = [1, 2, 3, 5, 10]
    
        self.cfg = train_cfg
        print("init policy runner")
        self.policy_runner = InnerOnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device, history=self.history)

        print("init geom runner")
        self.alg_cfg = train_cfg["algorithm_geom"]
        self.policy_cfg = train_cfg["policy_geom"]
        self.device = device
        self.env = env
        self.num_geom_joints = int(torch.sum(env.get_geom_map()).item())
        # obs, extras = self.env.get_observations() -> make this to be a function of the inner policy runner
        # num_obs = env.num_envs
        # num_actions = int(self.num_geom_joints + (self.num_geom_joints * (self.num_geom_joints + 1)) / 2.0) # mean and L_cholskiy 
        self.num_actions = int(self.num_geom_joints) # only mean
        num_obs_train = len(self.policy_runner.needed_observations)
        num_obs_geom = self.num_geom_joints

        # self.num_obs = (num_obs_train * 1  + self.num_actions * 1) * (1+len(self.history))
        self.num_obs = 78
        # check for extra obs here, but not relevat for now
        num_critic_obs = self.num_obs    # TODO add extra obs (infos about ongoing training)here
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        # the action here is replacing the gradient, e.g. one for each mean and one for each L_cholskiy entry
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            self.num_obs, num_critic_obs, self.num_actions, **self.policy_cfg
        ).to(device)
        alg_class = eval(self.alg_cfg.pop("class_name")) # PPO
        self.alg: PPO = alg_class(actor_critic, device=device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env_geom"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization_geom"]
        if self.empirical_normalization:    # investigate what this is, keep it to false for now
            self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # init storrage and model
        self.alg.init_storage(
            env.num_envs,
            self.num_steps_per_env,
            [self.num_obs],
            [num_critic_obs],
            [self.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.current_learning_iteration = self.policy_runner.current_learning_iteration # I have to think how to handle this, it makes no sense to do doppelte buchführung, beter just use the iteration from the policy runner, in this case TODO delete this line 
        self.git_status_repos = [rsl_rl.__file__] # not used jet, but TODO implement (move this to outer loop)

        # Inintialize the geom distribution
        # mean and L_cholskiy
        self.mean = self.env.get_distruibution()[0].mean
        self.L_cholseky = torch.linalg.cholesky(self.env.get_distruibution()[0].covariance_matrix)


    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        ''' 

        '''
        print("learn geom")
        # why is this not in the init function?
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        # make some randomization of "episode length" here (just because at the inner loop it is at this position)

        # init observations n stuff
        # obs, critic_obs = self.policy_runner.env.get_observations()
        obs = critic_obs = torch.zeros(self.env.num_envs, 8, dtype=torch.float, device=self.device) # TODO:
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        actbuffer = deque(maxlen=100)
        opsbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.policy_runner.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.policy_runner.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        actions = torch.zeros(self.env.num_envs, self.num_actions, dtype=torch.float, device=self.device)

        num_policy_it = 50
        for it in range(start_iter, tot_iter):
            # with torch.inference_mode():
            # adapt the geometry
            if it > num_policy_it:
                actions = self.alg.act(obs, critic_obs)
                self.apply_actions(actions)
            print("it_outer loop", it)
            # sample data
            obs, rewards, dones, infos = self.policy_runner.step(self.writer, it)
            # logg history of actions and obs
            actbuffer.append(actions)
            opsbuffer.append(obs)
            # add history to obs
            obs = torch.cat((obs, actions.clone()), dim=1)
            for h in self.history:
                if h < len(opsbuffer):
                    obs = torch.cat((obs, opsbuffer[h].clone()), dim=1)
                    # print("dim opsbuffer", opsbuffer[h].shape)
                else: 
                    obs = torch.cat((obs, torch.zeros(self.env.num_envs, self.num_obs, device=self.device)), dim=1)
                if h < len(actbuffer):
                    # print(" dim actbuffer", actbuffer[h].shape)
                    obs = torch.cat((obs, actbuffer[h].clone()), dim=1)
                else:
                    obs = torch.cat((obs, torch.zeros(self.env.num_envs, self.num_actions, device=self.device)), dim=1)
                # print("dim obs", obs.shape)
            critic_obs = obs
            if it > num_policy_it:
                it = self.policy_runner.current_learning_iteration
                obs, critic_obs, rewards, dones = (
                            obs.to(self.device),
                            critic_obs.to(self.device),
                            rewards.to(self.device),
                            dones.to(self.device),
                        )
                obs = self.obs_normalizer(obs)
                # if "critic" in infos["observations"]:
                #     critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                # else:
                #     critic_obs = obs TODO:

                self.alg.process_env_step(rewards, dones, infos)

            if self.log_dir is not None:
                if "episode" in infos:
                    ep_infos.append(infos["episode"])
                elif "log" in infos:
                    ep_infos.append(infos["log"])
                cur_reward_sum += rewards
                cur_episode_length += 1
                new_ids = (dones > 0).nonzero(as_tuple=False)
                rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0
            
            if it > num_policy_it:
                self.alg.compute_returns(critic_obs)
            
                mean_value_loss, mean_surrogate_loss = self.alg.update()
            else:
                mean_value_loss, mean_surrogate_loss = 0, 0

            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)


        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            
            # log
            # save
            # update learning iteration



        # pass the logger to the policy runner

    def apply_actions_cholsky(self, actions):
        # this is a version, where the geom is represented by mean and L_cholsky
        # geom action dim = int(self.num_geom_joints + (self.num_geom_joints * (self.num_geom_joints + 1)) / 2.0)
        print("apply actions")
        # put noise (actions) on mother distribution and send distributions to env
        distributions = []
        lower_triangular_indices = torch.tril_indices(self.num_geom_joints, self.num_geom_joints)
        
        for i, action in enumerate(actions):
            L_cholseky_noise = torch.zeros_like(self.L_cholseky)
            L_cholseky_noise[lower_triangular_indices[0], lower_triangular_indices[1]] = action[len(self.mean):] * self.scale_cholseky
            noisy_L_cholseky = self.L_cholseky + L_cholseky_noise
            cov_matrix = torch.mm(noisy_L_cholseky, noisy_L_cholseky.t())
            # check if the matrix is positive definite
            eigenvalues = torch.linalg.eigvalsh(cov_matrix)
            if (eigenvalues >= 1e-8).all():
                noisy_mean = self.mean + action[:len(self.mean)] * self.scale_mean
                distributions.append(MultivariateNormal(noisy_mean, cov_matrix))
            else:
                distributions.append(MultivariateNormal(self.mean, self.L_cholseky))
                print("Matrix is not positive definite-----------------------------------")
        self.policy_runner.env.distribution_update(distributions)

        # calculate new mother distribution by adding the mean action to the old 
        # TODO limit the gradient somehow
        self.mean += actions[:, :len(self.mean)].mean(dim=0)
        self.L_cholseky[lower_triangular_indices[0], lower_triangular_indices[1]] += actions[:, len(self.mean):].mean(dim=0) # TODO, when add comands for ppo, limit lenth here
        # TODO check if mother distribution is positive definite
        # TODO apply actions to the inner loop

    def apply_actions(self, actions):
        distributions = []
        for i, action in enumerate(actions):
            # print("action", action)
            mean = torch.ones_like(self.mean) * 0.5 + torch.tanh(action) * 0.5
            # print("action_tanh", torch.tanh(action))
            distributions.append(MultivariateNormal(mean.to(self.device), torch.eye(self.num_geom_joints).to(self.device) * 1e-8))
        self.policy_runner.env.distribution_update(distributions)

        print("mean",torch.mean(actions, dim=0))

        self.mean = torch.ones_like(self.mean) * 0.5 + torch.tanh(torch.mean(actions,dim=0)) * 0.5
        print("mean", self.mean)
        self.L_cholseky = torch.eye(self.num_geom_joints) * 1e-8   

    def save(self, path, infos=None):
        save_dict = {
            "policy": {
                "model_state_dict": self.policy_runner.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.policy_runner.alg.optimizer.state_dict(),
                "iter": self.policy_runner.current_learning_iteration,
                "infos": infos["policy"] if infos is not None and "policy" in infos else None,
            },
            "geometry": {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "mean": self.mean,
                "L_cholseky": self.L_cholseky,
                "infos": infos["geometry"] if infos is not None and "geometry" in infos else None,
            },
            "infos": infos,
        }
        if self.policy_runner.empirical_normalization:
            save_dict["policy"]["obs_normalizer"] = self.policy_runner.obs_normalizer.state_dict()
            save_dict["policy"]["critic_obs_normalizer"] = self.policy_runner.critic_obs_normalizer.state_dict()
        if self.empirical_normalization:
            save_dict["geometry"]["obs_normalizer"] = self.obs_normalizer.state_dict()
            save_dict["geometry"]["critic_obs_normalizer"] = self.critic_obs_normalizer.state_dict()
        torch.save(save_dict, path)
        print("saved")
        # uplaod model to extrnal logger
        if self.logger_type in ["wandb", "neptune"]:
            self.writer.save_model(path, self.current_learning_iteration)
        print("uploaded")


    def log(self, locals):
        # logg all the data from the outer runner, add outer to the names
        # self.tot_timesteps += self.num_steps_per_env * self.env.num_envs # TODO make this propper
        # self.tot_time += locals["collect_time"] + locals["learn_time"]
        # ....
        self.writer.add_scalar("Test_Log1", 1.0, locals["it"])

        mean_std = self.alg.actor_critic.std.mean()

        self.writer.add_scalar("Outer/Loss/value_function", locals["mean_value_loss"], locals["it"])
        self.writer.add_scalar("Outer/Loss/surrogate", locals["mean_surrogate_loss"], locals["it"])
        self.writer.add_scalar("Outer/Loss/learning_rate", self.alg.optimizer.param_groups[0]["lr"], locals["it"])
        self.writer.add_scalar("Outer/mean_std", mean_std.item(), locals["it"])
        if len(locals["rewbuffer"]) > 1:
            self.writer.add_scalar("Outer/mean_reward", statistics.mean(locals["rewbuffer"]), locals["it"])
            self.writer.add_scalar("Outer/std_reward", statistics.stdev(locals["rewbuffer"]), locals["it"])

        # add geometiy data
        for i, manin in enumerate(self.mean):
            self.writer.add_scalar(f"Outer/mean_geom_{i}", manin.item(), locals["it"])
            # logg the variance
            self.writer.add_scalar(f"Outer/mena_var_{i}", locals["actions"][:, i].var().item(), locals["it"])
        

        
        lower_triangular_indices = torch.tril_indices(self.num_geom_joints, self.num_geom_joints)
        for i, j in zip(lower_triangular_indices[0], lower_triangular_indices[1]):
            self.writer.add_scalar(f"Outer/L_cholseky_geom_{i}_{j}", self.L_cholseky[i, j].item(), locals["it"])
        self.writer.add_scalar("Test_Log2", 1.0, locals["it"])


        # TODO add string
        print(" outer loop log")

    def load(self, path, load_optimizer=True):
        lodaded_dict = torch.load(path, weights_only=True)
        self.policy_runner.load_from_outer_loop(lodaded_dict["policy"], load_optimizer)

        print("loaded dict[geometry][mean]", lodaded_dict["geometry"]["mean"])
        self.mean = lodaded_dict["geometry"]["mean"]
        self.L_cholseky = lodaded_dict["geometry"]["L_cholseky"]
        self.learning_iteration = lodaded_dict["geometry"]["iter"]

        self.alg.actor_critic.load_state_dict(lodaded_dict["geometry"]["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(lodaded_dict["geometry"]["obs_normalizer"])
            self.critic_obs_normalizer.load_state_dict(lodaded_dict["geometry"]["critic_obs_normalizer"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(lodaded_dict["geometry"]["optimizer_state_dict"])
        self.current_learning_iteration = lodaded_dict["geometry"]["iter"]
        return lodaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.alg.actor_critic.to(device)

        if self.cfg["empirical_normalization_geom"]: # TODO check if this is the right way to do it
            if device is not None:
                self.obs_normalizer.to(device)
                self.critic_obs_normalizer.to(device)
        
        # create ne distributions to push to the env
        distributions = []
        cov_matrix = torch.eye(self.num_geom_joints) * 1e-8
        
        distributions = [MultivariateNormal(self.mean.to(self.device), cov_matrix.to(self.device))] * self.env.num_envs
        self.policy_runner.env.distribution_update(distributions)
        print("mean", self.mean)
        print(" pushed new geom distributions to env")
        # self.env.reset() # This would reset the whole env. but I only want to resample the geom as if it was from a timeout

        return self.policy_runner.get_inference_policy(device)


    def train_mode(self): # TODO dose that make sense this way'
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()


    def eval_mode(self): # TODO dose that make sense this way'
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_path):
        self.policy_runner.add_git_repo_to_log(repo_path)