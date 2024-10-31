import torch

from rsl_rl.env import VecEnv

# geometry clase here TODO: move to own file
# ________________________________________________________________________________________________________
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

        # Values to cntrol the geom update frequency
        self.min_it = 150
        self.it_interval = 5

        self.logged_itterations = torch.tensor([], device=self.device)


    def initialize_distributions(self):
        """
            initialize the distributions for the geometric joints
            return: 
                    shape: (num_joints, 2)
        """
        geom_distributions = torch.full((len(self.geom_mask), 2), float('nan'), device=self.device)
        for i, joint in enumerate(self.geom_mask):
            if joint == 1:
                geom_distributions[i] = torch.tensor([0.15, 0.5], device=self.device)
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
        if it > self.min_it and it % self.it_interval == 0:
            # estimate the gradient
            gradient = self.estimate_gradient()

            # clip the gradient
            if gradient > 0.1:
                gradient = 0.1

            # update the distribution
            self.distribution[:, 0] += gradient
            # clip the mean
            self.distribution[:, 0] = torch.clamp(self.distribution[:, 0], 0, 1)

    def estimate_gradient(self):
        """
            estimate gradinet of reward(geom)
        """
        # get the average value for each geometric joint
        geom = self.geomety
        average_geom = torch.mean(geom, dim=0)
        noise = geom - average_geom

        # print average_geom
        print(average_geom)
        # get the number of datapoints
        N = self.rewards.numel()

        # init gradient shape like self.mask
        gradient = torch.zeros_like(self.geom_mask, device=self.device)
        # caluculate the gradient
        for i in range(self.rewards.size(0)):
            for j in range(self.rewards.size(1)):
                gradient += noise[i, :] * self.rewards[i, j]
        gradient /= N
        return gradient


    def store_reward(self, reward, it):
        """
            store the reward in the buffer 
        """
        # TODO: could maybe use the rebuffer here
        

        # check if the reward needs to be logged
        if it > self.min_it:
            # check if the reward needs to be reset
            if self.logged_itterations.numel() >= self.it_interval:
                self.logged_itterations = torch.tensor([], device=self.device)
                self.rewards = torch.tensor([], device=self.device)
            if self.logged_itterations.size() == 0:
                self.logged_itterations = torch.tensor([it], device=self.device)
                self.rewards = reward
            elif self.logged_itterations[-1] == it:
                self.rewards[-1] += reward
            elif self.logged_itterations[-1] == it - 1:
                self.logged_itterations = torch.cat((self.logged_itterations, torch.tensor([it], device=self.device)))
                self.rewards = torch.cat((self.rewards, torch.tensor([reward], device=self.device)), dim=0)
    

    def update_geom(self, it):
        """
            sample values from the distribution and update the geometry values in the environment
        """
        # check if neccesary 
        if it % self.it_interval == 0:
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