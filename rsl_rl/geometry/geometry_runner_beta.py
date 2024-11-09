import torch
from torch.distributions import Normal, Beta


from rsl_rl.env import VecEnv

# geometry clase here TODO: move to own file
class GeometryRunnerBeta:
    def __init__(self, env: VecEnv, device: torch.device, steps_per_it):
        """Initializes the geometry runner.

        Args:
            env: The environment to interact with.
            device: The device to use.
        """
        self.stdDev_beta = 1.0

        self.env = env
        self.device = device

        # Initialize reward buffer and observation buffer
        # self.rewards = torch.tensor([], device=self.device)
        # initaialize rewards tensor to be size of num evns
        self.rewards = torch.empty((0, env.num_envs), device=self.device)
        self.geometry_log = torch.tensor([], device=self.device)
        self.distribution_log = torch.tensor([], device=self.device)

        # Get geom mask
        self.geom_mask = self.env.get_geom_map()
        self.distribution = self.initialize_distributions()
        self.geomety = self.initialize_geometry()

        # Values to cntrol the geom update frequency
        self.min_it = 150
        self.policy_it = 0
        self.goem_it = 5
        self.it_interval = self.policy_it + self.goem_it
        self.steps_per_it = steps_per_it
        self.count_steps = 0


        self.logged_itterations = 0
        self.last_itteration = 0


        self.gradient = torch.tensor([0.0, 0., 0., 0., 0.], device=self.device)
        self.gradient_analytic = torch.tensor([0.0], device=self.device)

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
                geom_distributions[i] = torch.tensor([20.0, 20.0], device=self.device)
        # fill the distribution log
        self.fill_distribution_log(geom_distributions)

        # print("geom_distributions after initalisateion", geom_distributions)
        return geom_distributions
    
    def fill_distribution_log(self, geom_distributions = None):
        """
            Fills the distribution log with beta distributions for each environment.
        """
        if geom_distributions is None:
            geom_distributions = self.distribution
        num_envs = self.env.num_envs
        num_joints = len(self.geom_mask)
        distribution_log = torch.full((num_envs, num_joints, 2), float('nan'), device=self.device)

        for env_idx in range(num_envs):
            for joint_idx in range(num_joints):
                if self.geom_mask[joint_idx] == 1:
                    alpha = geom_distributions[joint_idx, 0]
                    beta = geom_distributions[joint_idx, 1]

                    # Add Gaussian noise to alpha and beta
                    noise_dist = Normal(0, self.stdDev_beta)
                    alpha_noisy = alpha + noise_dist.sample()
                    beta_noisy = beta + noise_dist.sample()

                    # Ensure alpha and beta are positive
                    alpha_noisy = torch.clamp(alpha_noisy, min=1)
                    beta_noisy = torch.clamp(beta_noisy, min=1)

                    # Store the noisy alpha and beta in the distribution log
                    distribution_log[env_idx, joint_idx] = torch.tensor([alpha_noisy, beta_noisy], device=self.device)

        self.distribution_log = distribution_log
        # print("distribution_log", distribution_log)
    
    def initialize_geometry(self):
        """
            initialize the geometry values for the geometric joints
            return: 
                    shape: (num_envs, num_joints)
        """
        # Create a geometry tensor based on the mask, repeated for each environment
        # geom = self.geom_mask.clone().float().unsqueeze(0).repeat(self.env.num_envs, 1)
        geom = self.geom_mask.clone().float().masked_fill_(self.geom_mask == 0, float('nan')).unsqueeze(0).repeat(self.env.num_envs, 1)

        # print("geom_before initialisation", geom)
        # print("geom_mask", self.geom_mask)
        for i in range(len(self.geom_mask)):
            # if not torch.isnan(self.distribution_log[0, i, 0]):
            if self.geom_mask[i] == 1:
                # Sample from the noisy beta distribution for each environment
                beta_samples = Beta(self.distribution_log[:, i, 0], self.distribution_log[:, i, 1]).sample()
                geom[:, i] = beta_samples
        # print("geom_after initialisation", geom)
        self.geomety = geom
        return geom

    def update_distributions(self, it):
        """
            process for updating the distributions from observations and rewards
        """
        gradient = self.estimate_gradient()
        # might be interisting to calculate also the gradient of the geometry
        #     # clip the gradient to [-0.1, 0.1]
        #     gradient = torch.clamp(gradient, -0.1, 0.1)

        #     # create nomal distributions
        #     distributions = Normal(loc=self.distribution[:, 0], scale=self.distribution[:, 1])
        #     z_values = distributions.icdf((1+self.p)/2)
        #     variance_update = torch.sign(gradient) * (abs(gradient)-z_values**2)

        #     print("distributions", distributions)
        #     print("z_values", z_values)
        #     print("gradient", gradient)
        #     print("variance_update", variance_update)
        #     print("p", self.p)
        #     # update the distribution
        #     self.distribution[:, 0] += gradient
        #     # self.distribution[:, 1] += variance_update

        #     # print("gradient", gradient)
        #     # print("z_values", z_values)
        #     # print("variance_update", variance_update)

        #     # clip the mean
        #     self.distribution[:, 0] = torch.clamp(self.distribution[:, 0], 0, 1)
        print("gradient", gradient)
        gradient = torch.clamp(gradient, -2.0, 2.0)
        self.distribution += gradient
        print("gradient", gradient)
        # clip the beta distribution to [1.0, inf]
        self.distribution = torch.clamp(self.distribution, 1., 1000.)
        # fill the distribution log
        self.fill_distribution_log(self.distribution)
        self.distribution_log = torch.clamp(self.distribution_log, 1., 1000.)

    def estimate_gradient(self):
        """
            estimate gradient of reward(geom)
        """
        # print("self.geometry_log", self.geometry_log)
        print("self.dist", self.distribution)
        print("self.distribution_log", self.distribution_log)

        # get the average value for each geometric joint
        geom = self.geometry_log
        average_geom = torch.mean(torch.mean(geom, dim=0), dim=0)

        
        # get the average value for each distribution parameter TODO: use the nois calculated when filling the distribution log
        dist = self.distribution_log # shape: (num_envs, num_joints, 2)
        average_dist = torch.mean(dist, dim=0) # shape: (num_joints, 2)
        noise = dist - average_dist.unsqueeze(0)  # shape: (num_envs, num_joints, 2)

        gradient = torch.zeros_like(self.distribution, device=self.device)  # shape: (num_joints, 2)

        # print the dimesions
        # print("dist", dist.size())
        # print("average_dist", average_dist.size())
        # print("noise", noise.size())
        # print("rewards", self.rewards.size())
        # print("reward", self.rewards)

        # print("still alive")
        # calculate the gradient for beta distributions
        N = self.rewards.numel()
        for i in range(self.rewards.size(0)):
            gradient[:, 0] += noise[i, :, 0] * self.rewards[i]  
            gradient[:, 1] += noise[i, :, 1] * self.rewards[i] 
        # print("survied until here")
        gradient /= N
        self.gradient = gradient

        return gradient


    def store_reward(self, reward, it):
        
        if it > self.min_it:
        # print("it", it)
        # print("self.logged_itterations", self.logged_itterations)
        # print("self.last_itteration", self.last_itteration)
            # if self.rewards.size != reward.size:
            #     self.rewards = torch.empty_like(reward, device=self.device)
            #     # self.rewards = torch.empty((0, reward.size(0)), device=self.device)
            #     print("initailized rewards")
            #     print("reward", reward)
            #     print("self.rewards", self.rewards)

            if self.logged_itterations > (self.policy_it)  or self.policy_it == 0 or (self.logged_itterations == (self.policy_it) and self.count_steps == self.steps_per_it):
                # print("logge geometry: logge itterations", self.logged_itterations, "last itteration", self.last_itteration, "it", it)
                if self.last_itteration == it:
                    # print("point_1")
                    # logg the reward
                    self.rewards += reward
                    # check if we need to update the distributions
                    # print("point_2")
                    self.count_steps += 1
                    if self.rewards.numel() == 0:
                        self.rewards = reward.clone()
                    else:
                        self.rewards += reward

                    if self.logged_itterations > self.it_interval and self.count_steps == self.steps_per_it:
                        # print("point_3")
                        # reset everything
                        self.update_distributions(it)
                        self.rewards = torch.empty((0, reward.size(0)), device=self.device)
                        self.geometry_log = torch.tensor([], device=self.device)
                        self.logged_itterations = 0
                        # self.last_itteration = it
                        self.count_steps = 0
                # elif self.last_itteration == it - 1: # do we have a new it value?
                elif self.last_itteration < it:
                    # print("point_4")
                    # create new itteration in logging values
                    self.geometry_log = torch.cat((self.geometry_log, self.geomety.unsqueeze(0).clone()), dim=0)    # logg the geometry
                    if self.rewards.numel() == 0:
                        self.rewards = reward.clone()
                    else:
                        self.rewards += reward

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

    def update_geom(self, it):
        """
            sample values from the distribution and update the geometry values in the environment
        """
        
        self.initialize_geometry()
        # print("geomety_geom update", self.geomety)
        self.env.geom_update(self.geomety)

    def logger(self):
        """
            log the geometry values
        """

        geom = self.geomety
        average_geom = torch.mean(geom, dim=0)
        average_geom = average_geom[self.geom_mask == 1]


        return average_geom, self.gradient