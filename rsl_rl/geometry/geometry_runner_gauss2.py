import torch

from rsl_rl.env import VecEnv

# geometry clase here TODO: move to own file
class GeometryRunnerGauss:
    def __init__(self, env: VecEnv, device: torch.device, steps_per_it, min_it, policy_it, goem_it):
        """Initializes the geometry runner.

        Args:
            env: The environment to interact with.
            device: The device to use.
        """
        self.env = env
        self.device = device

        # Get geom mask
        self.geom_mask = self.env.get_geom_map()
        self.distribution = self.initialize_distributions()
        self.geomety = self.initialize_geometry()

        # Initialize reward buffer and observation buffer
        self.rewards = torch.tensor([], device=self.device)
        self.geometry_log = torch.tensor([], device=self.device)
        self.specific_reward = torch.tensor([], device=self.device)

        # Values to cntrol the geom update frequency
        self.min_it = min_it
        self.policy_it = policy_it
        self.goem_it = goem_it
        self.it_interval = self.policy_it + self.goem_it
        self.steps_per_it = steps_per_it
        self.count_steps = 0


        self.logged_itterations = 0
        self.last_itteration = 0


        self.gradient = torch.tensor([0.0, 0., 0., 0., 0.], device=self.device)
        self.gradient_analytic = torch.tensor([0.0], device=self.device)
        self.specific_gradient = torch.tensor([0.0, 0., 0., 0., 0.], device=self.device)

        self.g = 0


    def initialize_distributions(self):
        """
            initialize the distributions for the geometric joints
            return: 
                    shape: (num_joints, 2)
        """
        geom_distributions = torch.full((len(self.geom_mask),2), float('nan'), device=self.device)
        for i, joint in enumerate(self.geom_mask):
            if joint == 1:
                geom_distributions[i] = torch.tensor([0.5, 0.15], device=self.device)
        return geom_distributions
    
    def initialize_geometry(self):
        """
            initialize the geometry values for the geometric joints
            return: 
                    shape: (num_envs, num_joints)
        """
        # Create a geometry tensor based on the mask, repeated for each environment
        geom = self.geom_mask.clone().float().unsqueeze(0).repeat(self.env.num_envs, 1)

        # Iterate over each joint and set geometry values for non-filtered joints
        for i, dist in enumerate(self.distribution):
            if not torch.isnan(dist[0]):
                # Create Gaussian distribution with mean and std for the corresponding joint
                gaussian_samples = torch.normal(mean=dist[(-1) ].item(), std=dist[1].item(), size=(self.env.num_envs,), device=self.device)
                geom[:, i] = gaussian_samples
        
        return geom

    def update_distributions(self, it, specific_reward = None):
        """
            process for updating the distributions from observations and rewards
        """
        gradient = self.estimate_gradient(specific_reward)

        # clip the gradient to [-0.1, 0.1]
        gradient = torch.clamp(gradient, -0.1, 0.1)

        # update the distribution
        self.distribution[:, 0] += gradient
        # clip the mean
        self.distribution[:, 0] = torch.clamp(self.distribution[:, 0], 0, 1)

    def estimate_gradient(self, specific_reward = None):
        """
            estimate gradient of reward(geom)
        """
        # get the average value for each geometric joint
        geom = self.geometry_log
        average_geom = torch.mean(torch.mean(geom, dim=0), dim=0)
        noise = geom - average_geom
        N = self.rewards.numel()

        # print the dimesions
        # print("geom", geom.size())
        # print("average_geom", average_geom.size())
        # print("noise", noise.size())
        # print("rewards", self.rewards.size())


        # init gradient shape like self.mask    
        gradient = torch.zeros_like(self.geom_mask, device=self.device)
        # calculate the gradient
        h = 0
        for i in range(self.rewards.size(0)):
            for j in range(self.rewards.size(1)):
                gradient += noise[i, j] * self.rewards[i, j]
                h += 1
        gradient /= N
        self.gradient = gradient

        if specific_reward is not None:
            # calculate analytic gradient under the assumption, that pole is vertical
            w = -5
            h_goal = 1
            dt = 1/120 #0.016666666666666666 # try out 1 / 120 (set in env. config)
            lenth = average_geom[self.geom_mask == 1].item()
            # gradient_analytic = - 2 * w * (h_goal - average_geom[2]) * dt
            gradient_analytic = - 2 * w * (h_goal - lenth) * dt
            # print("average_geom", average_geom)
            # print("avergae_geom[1]", average_geom[2])
            # print("gradient_analytic", gradient_analytic)
            # print("geom", geom)
            # print("noise", noise)
            self.gradient_analytic = torch.tensor(gradient_analytic, device=self.device)

            # calculate stochastic gradient only on specific reward
            N_specific = specific_reward.numel()
            specific_gradient = torch.zeros_like(self.geom_mask, device=self.device)
            print("self.rewards", self.rewards)
            print("specific_reward", self.specific_reward)
            for i in range(self.specific_reward.size(0)):
                for j in range(self.specific_reward.size(1)):
                    specific_gradient += noise[i, j] * self.specific_reward[i, j]
            specific_gradient /= N_specific
            self.specific_gradient = specific_gradient


        print("factor", self.gradient_analytic / self.gradient)




        return gradient


    def store_reward(self, reward, it, specific_reward = None):
        # print("specific_reward", specific_reward)

        if it > self.min_it:
            # print("it", it)
            # print("self.logged_itterations", self.logged_itterations)
            # print("self.last_itteration", self.last_itteration)

            if self.logged_itterations > (self.policy_it)  or self.policy_it == 0 or (self.logged_itterations == (self.policy_it) and self.count_steps == self.steps_per_it):
                # print("logge geometry: logge itterations", self.logged_itterations, "last itteration", self.last_itteration, "it", it)
                if self.last_itteration == it:
                    # print("point_1")
                    # logg the reward
                    self.rewards[-1, :] += reward
                    if specific_reward is not None:
                        self.specific_reward[-1, :] += specific_reward
                    # check if we need to update the distributions
                    # print("point_2")
                    self.count_steps += 1
                    if self.logged_itterations > self.it_interval and self.count_steps == self.steps_per_it:
                        # print("point_3")
                        # reset everything
                        self.update_distributions(it, specific_reward)
                        self.rewards = torch.empty((0, reward.size(0)), device=self.device)
                        if specific_reward is not None:
                            self.specific_reward = torch.empty((0, specific_reward.size(0)), device=self.device)
                        self.geometry_log = torch.tensor([], device=self.device)
                        self.logged_itterations = 0
                        # self.last_itteration = it
                        self.count_steps = 0
                # elif self.last_itteration == it - 1: # do we have a new it value?
                elif self.last_itteration < it:
                    # print("point_4")
                    # create new itteration in logging values
                    self.geometry_log = torch.cat((self.geometry_log, self.geomety.unsqueeze(0).clone()), dim=0)
                    self.rewards = torch.cat((self.rewards, reward.unsqueeze(0)), dim=0)
                    if specific_reward is not None:
                        self.specific_reward = torch.cat((self.specific_reward, specific_reward.unsqueeze(0)), dim=0)
                    # update logging counter
                    self.logged_itterations += 1
                    self.last_itteration = it
                    self.count_steps = 1
                else:
                    print ("error-----------------------------------------------------------------------------------") # Handle somehow different
                return True
            else:
                # print("point_5")
                self.count_steps += 1
                if self.last_itteration < it:
                    # print("point_6")
                    self.logged_itterations += 1
                    self.last_itteration = it
                    self.count_steps = 1
                return False

    def update_geom(self, it, specific_reward = None):
        """
            sample values from the distribution and update the geometry values in the environment
        """
        # sample values from the distribution
        for i, dist in enumerate(self.distribution):
            if not torch.isnan(dist[0]):
                # Create Gaussian distribution with mean and std for the corresponding joint
                gaussian_samples = torch.normal(mean=dist[0].item(), std=dist[1].item(), size=(self.env.num_envs,), device=self.device)
                # clip distibution valuse [0, 1]
                gaussian_samples = torch.clamp(gaussian_samples, 0, 1)
                self.geomety[:, i] = gaussian_samples

            # update the geometry values in the environment
        self.env.geom_update(self.geomety)

    def logger(self):
        """
            log the geometry values
        """

        geom = self.geomety
        average_geom = torch.mean(geom, dim=0)
        average_geom = average_geom[self.geom_mask == 1]


        return average_geom, self.gradient