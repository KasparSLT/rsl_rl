import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.env import VecEnv

# Neural Network for function estimation
class FunctionEstimator(nn.Module):
    def __init__(self, input_size, output_size, steps_per_it, min_it, policy_it, goem_it):
        super(FunctionEstimator, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# GeometryRunner with Neural Network
class NNGeometryRunner:
    def __init__(self, env: VecEnv, device: torch.device):
        """Initializes the geometry runner with a neural network.

        Args:
            env: The environment to interact with.
            device: The device to use.
        """
        self.env = env
        self.device = device

        # Get geom mask
        self.geom_mask = self.env.get_geom_map()
        self.distribution = self.initialize_distributions()
        self.geometry = self.initialize_geometry()

        # Initialize reward buffer and observation buffer
        self.rewards = torch.tensor([], device=self.device)
        self.geometry_log = torch.tensor([], device=self.device)

        # Values to control the geom update frequency
        self.min_it = 150
        self.it_interval = 5

        self.logged_iterations = torch.tensor([], device=self.device)

        # Neural network for function estimation
        self.model = FunctionEstimator(input_size=len(self.geom_mask), output_size=1).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def initialize_distributions(self):
        """
            Initialize the distributions for the geometric joints
            return: 
                    shape: (num_joints, 2)
        """
        geom_distributions = torch.full((len(self.geom_mask), 2), float('nan'), device=self.device)
        for i, joint in enumerate(self.geom_mask):
            if joint == 1:
                geom_distributions[i] = torch.tensor([0.5, 0.15], device=self.device)
        return geom_distributions
    
    def initialize_geometry(self):
        """
            Initialize the geometry values for the geometric joints
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
            Process for updating the distributions from observations and rewards
        """
        # Estimate the gradient using the neural network
        gradient = self.estimate_gradient()

        # Clip the gradient to [-0.1, 0.1]
        gradient = torch.clamp(gradient, -0.1, 0.1)

        # Update the distribution
        self.distribution[:, 0] += gradient
        # Clip the mean
        self.distribution[:, 0] = torch.clamp(self.distribution[:, 0], 0, 1)

    def estimate_gradient(self):
        """
            Estimate gradient of reward(geom) using the neural network
        """
        self.model.train()

        geom = self.geometry
        rewards = self.rewards.unsqueeze(1)  # Make rewards a column vector

        # Forward pass through the model to get predictions
        predictions = self.model(geom)

        # Calculate loss
        loss = self.criterion(predictions, rewards)

        # Backward pass to calculate gradients
        self.optimizer.zero_grad()
        loss.backward()

        # Extract gradients for the input layer
        with torch.no_grad():
            gradient = self.model.fc1.weight.grad.mean(dim=0)
        
        return gradient

    def store_reward(self, reward, it):
        """
            Store the reward in the buffer 
        """
        if it > self.min_it:
            if self.logged_iterations.numel() >= self.it_interval:
                self.update_distributions(it)
                self.logged_iterations = torch.tensor([], device=self.device)
                self.rewards = torch.empty((0, reward.size(0)), device=self.device)
                self.geometry_log = torch.tensor([], device=self.device)

            if self.logged_iterations.numel() == 0:
                self.logged_iterations = torch.tensor([it], device=self.device).unsqueeze(0)
                self.rewards = reward.clone().unsqueeze(0)
            elif self.logged_iterations[-1] == it:
                self.rewards[-1, :] += reward
            elif self.logged_iterations[-1] == it - 1:
                self.logged_iterations = torch.cat((self.logged_iterations, torch.tensor([[it]], device=self.device)), dim=0)
                self.rewards = torch.cat((self.rewards, reward.unsqueeze(0)), dim=0)

    def update_geom(self, it):
        """
            Sample values from the distribution and update the geometry values in the environment
        """
        if it % self.it_interval == 0:
            for i, dist in enumerate(self.distribution):
                if not torch.isnan(dist[0]):
                    gaussian_samples = torch.normal(mean=dist[0].item(), std=dist[1].item(), size=(self.env.num_envs,), device=self.device)
                    gaussian_samples = torch.clamp(gaussian_samples, 0, 1)
                    self.geometry[:, i] = gaussian_samples

            self.env.geom_update(self.geometry)

            if self.geometry_log.numel() == 0:
                self.geometry_log = self.geometry.clone()
            else:
                self.geometry_log = torch.cat((self.geometry_log, self.geometry.clone()), dim=0)

    def logger(self):
        """
            Log the geometry values
        """
        geom = self.geometry
        average_geom = torch.mean(geom, dim=0)
        average_geom = average_geom[self.geom_mask == 1]
        return average_geom, self.model.fc1.weight.grad.mean(dim=0) if self.model.fc1.weight.grad is not None else torch.zeros_like(self.geom_mask, device=self.device)
