import torch

from rsl_rl.env import VecEnv

# geometry clase here TODO: move to own file
class GeometryRunner:
    def __init__(self, env: VecEnv, device: torch.device):
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

        # Values to cntrol the geom update frequency
        self.min_it = 150
        self.it_interval = 5

        self.logged_itterations = torch.tensor([], device=self.device)

        self.gradient = torch.tensor([0.0, 0., 0., 0., 0.], device=self.device)

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
                gaussian_samples = torch.normal(mean=dist[0].item(), std=dist[1].item(), size=(self.env.num_envs,), device=self.device)
                geom[:, i] = gaussian_samples
        
        return geom

    def update_distributions(self, it):
        """
            process for updating the distributions from observations and rewards
        """
        # check if needed
        # if it > self.min_it and it % self.it_interval == 0:
        # estimate the gradient
        gradient = self.estimate_gradient()

        # clip the gradient to [-0.1, 0.1]
        gradient = torch.clamp(gradient, -0.1, 0.1)

        # update the distribution
        self.distribution[:, 0] += gradient
        # clip the mean
        self.distribution[:, 0] = torch.clamp(self.distribution[:, 0], 0, 1)

        # print all infos
        # print("Update distributions-----------------------------------------------------------------------------------------------")
        # print("reward: ", self.rewards)
        # print("geometries: ", self.geomety)
        # print("gradient: ", gradient)

    def estimate_gradient(self):
        """
            estimate gradient of reward(geom)
        """
        # get the average value for each geometric joint
        geom = self.geometry_log
        average_geom = torch.mean(geom, dim=0)
        noise = geom - average_geom

        # print average_geom
        # print("average", average_geom)
        # get the number of datapoints1
        N = self.rewards.numel()

        # print("geometry log", self.geometry_log)
        # print("noise", noise)
        # print("rewards", self.rewards)

        # init gradient shape like self.mask    
        gradient = torch.zeros_like(self.geom_mask, device=self.device)
        # calculate the gradient
        print("i", self.rewards.size(0))
        print("j", self.rewards.size(1))

        print("i_2", noise.size(0))
        print("j_2", noise.size(1))
        print("n", noise.size(2))

        for i in range(self.rewards.size(0)):
            for j in range(self.rewards.size(1)):
                gradient += noise[i, j] * self.rewards[i, j]
        gradient /= N
        self.gradient = gradient
        # print("gradient", gradient)
        return gradient


    def store_reward(self, reward, it):
        """
            store the reward in the buffer 
        """
        # TODO: could maybe use the rebuffer here
        

        # check if the reward needs to be logged
        if it > self.min_it:
            # check if the reward needs to be reset
            # print(self.rewards)
            # if (self.logged_itterations.numel()) >= self.it_interval and self.g == 15:
            if (self.logged_itterations.numel()) >= self.it_interval:
                self.update_distributions(it)
                self.logged_itterations = torch.tensor([], device=self.device)
                self.rewards = torch.empty((0, reward.size(0)), device=self.device)
                self.geometry_log = torch.tensor([], device=self.device)
                print("g", self.g)

            if self.logged_itterations.numel() == 0:
                self.logged_itterations = torch.tensor([it], device=self.device).unsqueeze(0)
                self.rewards = reward.clone().unsqueeze(0)
                # log the geometry values
                if self.geometry_log.numel() == 0:
                    self.geometry_log = self.geomety.unsqueeze(0).clone()
                else:
                    self.geometry_log = torch.cat((self.geometry_log, self.geomety.unsqueeze(0).clone()), dim=0)
                self.g = 0

            elif self.logged_itterations[-1] == it:
                # print("add reward")
                # print(reward)
                self.rewards[-1, :] += reward
                self.g += 1
            elif self.logged_itterations[-1] == it - 1:
                self.g = 0

                # print("add new reward")
                self.logged_itterations = torch.cat((self.logged_itterations, torch.tensor([[it]], device=self.device)), dim=0)
                self.rewards = torch.cat((self.rewards, reward.unsqueeze(0)), dim=0)
                # log the geometry values
                if self.geometry_log.numel() == 0:
                    self.geometry_log = self.geomety.unsqueeze(0).clone()
                else:
                    self.geometry_log = torch.cat((self.geometry_log, self.geomety.unsqueeze(0).clone()), dim=0)


    def update_geom(self, it):
        """
            sample values from the distribution and update the geometry values in the environment
        """
        # check if neccesary 
        if it % self.it_interval == 0 or it == 0:
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

            # # log the geometry values
            # if self.geometry_log.numel() == 0:
            #     self.geometry_log = self.geomety.unsqueeze(0).clone()
            # else:
            #     self.geometry_log = torch.cat((self.geometry_log, self.geomety.unsqueeze(0).clone()), dim=0)

    def logger(self):
        """
            log the geometry values
        """

        geom = self.geomety
        average_geom = torch.mean(geom, dim=0)
        average_geom = average_geom[self.geom_mask == 1]


        return average_geom, self.gradient