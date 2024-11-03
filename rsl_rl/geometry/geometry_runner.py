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

    def update_distributions(self, it):
        """
            process for updating the distributions from observations and rewards
        """
        gradient = self.estimate_gradient()

        # clip the gradient to [-0.1, 0.1]
        gradient = torch.clamp(gradient, -0.1, 0.1)

        # update the distribution
        self.distribution[:, 0] += gradient
        # clip the mean
        self.distribution[:, 0] = torch.clamp(self.distribution[:, 0], 0, 1)

    def estimate_gradient(self):
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

        # calculate analytic gradient under the assumption, that pole is vertical
        w = -5
        h_goal = 1
        dt = 0.016666666666666666
        lenth = average_geom[self.geom_mask == 1].item()
        # gradient_analytic = - 2 * w * (h_goal - average_geom[2]) * dt
        gradient_analytic = - 2 * w * (h_goal - lenth) * dt
        # print("average_geom", average_geom)
        # print("avergae_geom[1]", average_geom[2])
        # print("gradient_analytic", gradient_analytic)
        # print("geom", geom)
        # print("noise", noise)
        self.gradient_analytic = torch.tensor(gradient_analytic, device=self.device)
        print("factor", self.gradient_analytic / self.gradient)




        return gradient


    def store_reward(self, reward, it):
        """
            store the reward in the buffer 
        """
        # TODO: could maybe use the rebuffer here
        

        # check if the reward needs to be logged
        if it > self.min_it:
            # check if the reward needs to be reset
            # if (self.logged_itterations.numel()) >= self.it_interval and self.g == 15:
            if (self.logged_itterations.numel()) >= self.it_interval:
                self.update_distributions(it)
                self.logged_itterations = torch.tensor([], device=self.device)
                self.rewards = torch.empty((0, reward.size(0)), device=self.device)
                # self.update_geom(it)
                self.geometry_log = torch.tensor([], device=self.device)
                # print("update geom")

            if self.logged_itterations.numel() == 0:
                self.logged_itterations = torch.tensor([it], device=self.device).unsqueeze(0)
                self.rewards = reward.clone().unsqueeze(0)
                # log the geometry values
                if self.geometry_log.numel() == 0:
                    self.geometry_log = self.geomety.unsqueeze(0).clone()
                else:
                    self.geometry_log = torch.cat((self.geometry_log, self.geomety.unsqueeze(0).clone()), dim=0)
                self.g = 0
                # print("reward_1")

            elif self.logged_itterations[-1] == it:
                self.rewards[-1, :] += reward
                # print("reward_2")

            elif self.logged_itterations[-1] == it - 1:
                self.logged_itterations = torch.cat((self.logged_itterations, torch.tensor([[it]], device=self.device)), dim=0)
                self.rewards = torch.cat((self.rewards, reward.unsqueeze(0)), dim=0)
                # log the geometry values
                if self.geometry_log.numel() == 0:
                    self.geometry_log = self.geomety.unsqueeze(0).clone()
                else:
                    self.geometry_log = torch.cat((self.geometry_log, self.geomety.unsqueeze(0).clone()), dim=0)
                # print("reward_3")


    def update_geom(self, it):
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