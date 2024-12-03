import torch
from torch.distributions import MultivariateNormal
from rsl_rl.env import VecEnv

class GeometryRunnerMVN:
    def __init__(self, env: VecEnv, device: torch.device, steps_per_it, min_it, policy_it, goem_it):
        self.env = env
        self.device = device

        # Get the geometry mask from the environment
        self.geom_mask = self.env.get_geom_map()
        
        # Determine the number geometric joints
        self.num_joints = len(self.geom_mask[self.geom_mask == 1])

        # Initialize the geometry
        # Tensors for mena (size d) variances (size d) and covariances (size d(d-1)/2)
        # These are the distributions, keeping the individual parts sepparated for eayer handeling of the gradients
        self.mean = torch.full((self.num_joints,), 0.5, device=self.device)
        self.variances = torch.full((self.num_joints,), 0.05, device=self.device)
        self.covariances = torch.zeros((self.num_joints * (self.num_joints - 1)) // 2, device=self.device)
        
        # Initialize noise standard deviations for mean variances and covariances,
        self.mean_std = torch.full((self.num_joints,), 0.1, device=self.device)
        self.variances_std = torch.full((self.num_joints,), 0.0, device=self.device)
        self.covariances_std = torch.full(((self.num_joints * (self.num_joints - 1)) // 2,), 0.0, device=self.device)
        
        # Initialize gradient containers
        self.mean_gradient = torch.tensor([], device=self.device)
        self.variances_gradient = torch.tensor([], device=self.device)
        self.covariances_gradient = torch.tensor([], device=self.device)

        # Initialize logging containers
        self.rewards = torch.empty((0, env.num_envs), device=self.device)   # Log of rewards for each iteration
        self.geometry_log = torch.tensor([], device=self.device)    # Log of geometries for each iteration

        # Initialize storrage for the peturbed distributions
        # tensor with dimensions (num_envs, [[d],[d],[d(d-1)/2]]) = (num_envs, [[mean],[var],[cov]]),
        self.distribution_log = []  # List to store distributions for each environment

        self.geometry = torch.full((self.env.num_envs, len(self.geom_mask)), float('nan'), device=self.device)

        # Initialize noise log 
        self.mean_noise = []
        self.variances_noise = [] 
        self.covariances_noise = []

        # Initialize distributions and log
        self.initialize_distribution_log()
        self.sample_geometry()
        # Set up iteration-related variables
        self.min_it = min_it
        self.policy_it = policy_it
        self.goem_it = goem_it
        self.it_interval = self.policy_it + self.goem_it
        self.steps_per_it = steps_per_it

        self.mean_learning_rate = 1
        self.variance_learning_rate = 1
        self.covariance_learning_rate = 1

        self.count_steps = 0
        self.logged_iterations = 0
        self.last_iteration = 0
  

    def initialize_distribution_log(self):
        self.distribution_log = []  # Clear existing distributions

        for _ in range(self.env.num_envs):
            run = True
            while run:  # Loop until a valid covariance matrix is constructed
                # Sample Gaussian noise for mean based on `mean_std`
                mean_noise = torch.randn_like(self.mean) * self.mean_std
                noisy_mean = self.mean + mean_noise

                # Sample Gaussian noise for variances and covariances
                variance_noise = torch.randn_like(self.variances) * self.variances_std
                noisy_variances = torch.clamp(self.variances + variance_noise, min=1e-5)  # Ensure positive variances

                covar_noise = torch.randn_like(self.covariances) * self.covariances_std
                noisy_covariances = torch.clamp(self.covariances + covar_noise, min=-1.0, max=1.0)  # Clamp covariances

                # Construct the noisy covariance matrix
                noisy_cov_matrix = torch.diag(noisy_variances)
                off_diag_indices = torch.triu_indices(self.num_joints, self.num_joints, offset=1)
                noisy_cov_matrix[off_diag_indices[0], off_diag_indices[1]] = noisy_covariances

                # Symmetrize and add jitter
                noisy_cov_matrix = (noisy_cov_matrix + noisy_cov_matrix.T) / 2
                noisy_cov_matrix += torch.eye(self.num_joints, device=self.device) * 1e-4

                # Check eigenvalues
                eigenvalues = torch.linalg.eigvals(noisy_cov_matrix).real
                if (eigenvalues >= 1e-5).all():  # Valid matrix if all eigenvalues are positive
                    # store the noise
                    self.mean_noise.append(mean_noise)
                    self.variances_noise.append(variance_noise)
                    self.covariances_noise.append(covar_noise)

                    # Store the valid distribution
                    distribution = MultivariateNormal(noisy_mean, noisy_cov_matrix)
                    self.distribution_log.append(distribution)

                    run = False
                else:
                    print("Resampling noise due to invalid covariance matrix. Eigenvalues:", eigenvalues)




    def sample_geometry(self):
        # Sample a geometry from each distribution
        samples = torch.stack([d.sample() for d in self.distribution_log])
        self.geometry[:, self.geom_mask == 1] = samples

        # Clip the geometry to the valid range
        self.geometry = torch.clamp(self.geometry, 0, 1)

    def update_distributions(self):
        # self.estimate_gradient()
        # Update the mean, variances, and covariances
        self.mean += self.mean_gradient * self.mean_learning_rate
        self.variances += self.variances_gradient * self.variance_learning_rate
        self.covariances += self.covariances_gradient * self.covariance_learning_rate

        # clip the distributions where needed
        self.variances = torch.clamp(self.variances, 0.01, 10)
        self.covariances = torch.clamp(self.covariances, -1.0, 1.0)

    def estimate_gradient(self):
        # Compute the gradient of the mean, variace and covariance
        # print("rewards: ", self.rewards)
        # print("mean_noise: ", self.mean_noise)

        self.mean_gradient = torch.zeros(self.mean.size(), device=self.device)
        self.variances_gradient = torch.zeros(self.variances.size(), device=self.device)
        self.covariances_gradient = torch.zeros(self.covariances.size(), device=self.device)

        for i, reward in enumerate(self.rewards):
            self.mean_gradient += reward * self.mean_noise[i]
            self.variances_gradient += reward * self.variances_noise[i]
            self.covariances_gradient += reward * self.covariances_noise[i]
        
        self.mean_gradient /= self.rewards.numel()
        self.variances_gradient /= self.rewards.numel()
        self.covariances_gradient /= self.rewards.numel()

    def store_reward(self, reward, it):
        if it > self.min_it:
            # if self.logged_iterations >= self.policy_it or self.policy_it == 0:
            if self.logged_iterations > (self.policy_it)  or self.policy_it == 0 or (self.logged_iterations == (self.policy_it) and self.count_steps == self.steps_per_it):
                if self.last_iteration == it:
                    # self.rewards = torch.cat((self.rewards, reward.unsqueeze(0)), dim=0)
                    self.rewards += reward
                    self.count_steps += 1
                    if self.logged_iterations >= self.it_interval and self.count_steps == self.steps_per_it:
                        self.geom_opt_step()   # perform the whole geom opt step 
                        self.rewards = torch.empty((0, reward.size(0)), device=self.device)
                        self.geometry_log = torch.tensor([], device=self.device)
                        self.logged_iterations = 0
                        self.count_steps = 0
                elif self.last_iteration < it:
                    self.geometry_log = torch.cat((self.geometry_log, self.geometry.unsqueeze(0).clone()), dim=0)
                    # self.rewards = torch.cat((self.rewards, reward.unsqueeze(0)), dim=0)
                    if self.rewards.numel() == 0:
                        self.rewards = reward
                    else:
                        self.rewards += reward
                    self.logged_iterations += 1
                    self.last_iteration = it
                    self.count_steps = 1
                else:
                    print("Error in iteration tracking.")
                return True
            else:
                self.count_steps += 1
                if self.last_iteration < it:
                    self.logged_iterations += 1
                    self.last_iteration = it
                    self.count_steps = 1
                return False

    def update_geom(self, it, envs = None):
        if envs is None:
            envs = range(self.env.num_envs)
        self.sample_geometry()
        # print(f"Geometry: {self.geometry}")
        self.env.geom_update(self.geometry)

    def update_distribution(self, it, envs = None): 
        return ### TODO

    def geom_opt_step(self):
        self.estimate_gradient()
        self.update_distributions()
        self.initialize_distribution_log()

    def log(self, writer, locs):
        """
        Log the gradients and distributions using a writer (e.g., TensorBoard or wandb).
        """
        # Log the gradients
        if self.mean_gradient.numel() > 0:  # Check if gradients are initialized
            for i in range(self.mean_gradient.size(0)):  # Use .size(0) for the number of elements
                mean_grad = self.mean_gradient[i]
                if not torch.isnan(mean_grad):
                    writer.add_scalar(f"Gradient/mean_grad_{i}", mean_grad.item(), locs)

        if self.variances_gradient.numel() > 0:  # Check if variances gradients are initialized
            for i in range(self.variances_gradient.size(0)):
                variance_grad = self.variances_gradient[i]
                if not torch.isnan(variance_grad):
                    writer.add_scalar(f"Gradient/variances_grad_{i}", variance_grad.item(), locs)

        if self.covariances_gradient.numel() > 0:  # Check if covariances gradients are initialized
            for i in range(self.covariances_gradient.size(0)):
                covariance_grad = self.covariances_gradient[i]
                if not torch.isnan(covariance_grad):
                    writer.add_scalar(f"Gradient/covariances_grad_{i}", covariance_grad.item(), locs)

        # Log the distributions (mean, variances, covariances)
        if self.mean.numel() > 0:  # Check if distributions are initialized
            for i, mean in enumerate(self.mean):
                if not torch.isnan(mean):
                    writer.add_scalar(f"Distribution/mean_{i}", mean.item(), locs)

        if self.variances.numel() > 0:
            for i, variance in enumerate(self.variances):
                if not torch.isnan(variance):
                    writer.add_scalar(f"Distribution/variance_{i}", variance.item(), locs)

        if self.covariances.numel() > 0:
            for i, covariance in enumerate(self.covariances):
                if not torch.isnan(covariance):
                    writer.add_scalar(f"Distribution/covariance_{i}", covariance.item(), locs)


