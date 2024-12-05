import torch
from torch.distributions import MultivariateNormal
from rsl_rl.env import VecEnv

class GeometryRunnerMVN2:
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
        
        # Initialize noise standard deviations for mean variances and covariances, this is for exploration
        self.mean_std = torch.full((self.num_joints,), 0.1, device=self.device)
        self.variances_std = torch.full((self.num_joints,), 0.01, device=self.device)
        self.covariances_std = torch.full(((self.num_joints * (self.num_joints - 1)) // 2,), 0.01, device=self.device)

        # cholseky stuff
        self.L_cholseky = torch.linalg.cholesky(self.env.get_distruibution()[0].covariance_matrix)
        self.L_cholseky_grad = torch.zeros(self.L_cholseky.size(), device=self.device)
        # noise matrix for each env
        self.L_cholseky_noise = torch.zeros((self.env.num_envs, *self.L_cholseky.size()), device=self.device)
        self.L_cholseky_var_scale = 0.005
        self.L_cholseky_cov_scale = 0.005
        
        # Initialize gradient containers
        self.mean_gradient = torch.tensor([], device=self.device)
        self.variances_gradient = torch.tensor([], device=self.device)
        self.covariances_gradient = torch.tensor([], device=self.device)

        # Initialize logging containers
        self.rewards = torch.zeros(env.num_envs, device=self.device)   # Log of rewards for each iteration
        self.geometry_log = torch.tensor([], device=self.device)    # Log of geometries for each iteration

        # Initialize storrage for the peturbed distributions
        # tensor with dimensions (num_envs, [[d],[d],[d(d-1)/2]]) = (num_envs, [[mean],[var],[cov]]),
        self.distribution_log = []  # List to store distributions for each environment

        self.geometry = torch.full((self.env.num_envs, len(self.geom_mask)), float('nan'), device=self.device)

        self.number_of_samples_per_dist = torch.zeros(self.env.num_envs, device=self.device)
        self.min_samples_per_dist = 3 # must be grather than one

        # Initialize noise log 
        self.mean_noise = torch.zeros((self.env.num_envs, *self.mean.size()), device=self.device)
        self.variances_noise = torch.zeros((self.env.num_envs, *self.variances.size()), device=self.device)
        self.covariances_noise = torch.zeros((self.env.num_envs, *self.covariances.size()), device=self.device)

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

        self.first_log = True

        # Initialize distributions and log
        self.initialize_distribution
        self.initialize_distribution_log()
        self.sample_geometry(None, 0)

        
    def initialize_distribution(self):
        distribution = self.env.get_distruibution()
        self.mean = distribution.mean
        self.variances = distribution.variances
        self.covariances = distribution.covariances

    # def initialize_distribution_log(self):
    #     self.distribution_log = []  # Clear existing distributions
    #     print("create new distributions")
    #     print("mean", self.mean)
    #     print("variances", self.variances)
    #     print("covariances", self.covariances)

    #     for _ in range(self.env.num_envs):
    #         run = True
    #         while run:  # Loop until a valid covariance matrix is constructed  TODO: find a way to always construct valid covariance matrix
    #             # Sample Gaussian noise for mean based on `mean_std`
    #             mean_noise = torch.randn_like(self.mean) * torch.sqrt(self.mean_std)
    #             noisy_mean = self.mean + mean_noise

    #             # Sample Gaussian noise for variances and covariances
    #             variance_noise = torch.randn_like(self.variances) * torch.sqrt(self.variances_std)
    #             noisy_variances = torch.clamp(self.variances + variance_noise, min=1e-5)  # Ensure positive variances

    #             covar_noise = torch.randn_like(self.covariances) * torch.sqrt(self.covariances_std)
    #             noisy_covariances = torch.clamp(self.covariances + covar_noise, min=-1.0, max=1.0)  # Clamp covariances

    #             # Construct the noisy covariance matrix
    #             noisy_cov_matrix = torch.diag(noisy_variances) / 2
    #             off_diag_indices = torch.triu_indices(self.num_joints, self.num_joints, offset=1)
    #             noisy_cov_matrix[off_diag_indices[0], off_diag_indices[1]] = noisy_covariances

    #             # Symmetrize and add jitter
    #             noisy_cov_matrix = (noisy_cov_matrix + noisy_cov_matrix.T)
    #             noisy_cov_matrix += torch.eye(self.num_joints, device=self.device) * 1e-4
    #             # print("noisy_cov_matrix", noisy_cov_matrix)

    #             # Check eigenvalues
    #             eigenvalues = torch.linalg.eigvals(noisy_cov_matrix).real
    #             if (eigenvalues >= 1e-5).all():  # Valid matrix if all eigenvalues are positive
    #                 # store the noise
    #                 self.mean_noise.append(mean_noise)
    #                 self.variances_noise.append(variance_noise)
    #                 self.covariances_noise.append(covar_noise)

    #                 # Store the valid distribution
    #                 distribution = MultivariateNormal(noisy_mean, noisy_cov_matrix)
    #                 self.distribution_log.append(distribution)

    #                 run = False
    #             else:
    #                 print("Resampling noise due to invalid covariance matrix. Eigenvalues:", eigenvalues)

    def initialize_distribution_log(self): # this function is using cholesky decomposition to create a valid covariance matrix

        # Reset the loggs
        self.distribution_log = [] 
        self.L_cholseky_noise = torch.zeros((self.env.num_envs, *self.L_cholseky.size()), device=self.device)
        self.mean_noise = torch.zeros((self.env.num_envs, *self.mean.size()), device=self.device)
        self.variances_noise = torch.zeros((self.env.num_envs, *self.variances.size()), device=self.device)
        self.covariances_noise = torch.zeros((self.env.num_envs, *self.covariances.size()), device=self.device)

        # Calculate base covariance matrix for logging
        base_cov_matrix = torch.matmul(self.L_cholseky, self.L_cholseky.T)
        self.variance = torch.diag(base_cov_matrix)
        lower_triangular_indices = torch.tril_indices(self.num_joints, self.num_joints, offset=-1)
        self.covariances = base_cov_matrix[lower_triangular_indices[0], lower_triangular_indices[1]]


        for _ in range(self.env.num_envs):
            run = True
            while run:
                # Add Gaussian noise to diagonal elements (variances)
                L_diag_noise = torch.randn(self.L_cholseky.shape[0], device=self.device) * self.L_cholseky_var_scale
                cholseky_noise = self.L_cholseky + torch.diag(L_diag_noise)

                # Add noise to off-diagonal elements separately
                L_off_diag_noise = torch.randn_like(self.L_cholseky) * self.L_cholseky_cov_scale
                L_off_diag_noise = torch.tril(L_off_diag_noise, diagonal=-1)  # Keep only the lower triangular part
                cholseky_noise += L_off_diag_noise
                # Construct the noisy covariance matrix
                noisy_cov_matrix = torch.matmul(cholseky_noise, cholseky_noise.T)

                # Check if the matrix is positive definite
                eigenvalues = torch.linalg.eigvals(noisy_cov_matrix).real
                if (eigenvalues >= 1e-10).all():

                    # perturb the mean
                    mean_noise = torch.randn_like(self.mean) * self.mean_std
                    noisy_mean = self.mean + mean_noise

                    # Construct the noisy covariance matrix
                    self.distribution_log.append(MultivariateNormal(noisy_mean, noisy_cov_matrix))

                    # logg the noise
                    self.mean_noise[_] = mean_noise
                    self.L_cholseky_noise[_] = cholseky_noise

                    run = False
                else:
                    print("Resampling noise due to invalid covariance matrix. Eigenvalues:", eigenvalues)




    def sample_geometry(self, envs = None, it = 0): # TODO reomve
        if envs is None:
            envs = torch.ones(self.env.num_envs, device=self.device)
        # print("envs: ", envs)

        # check if each env has enough samples
        # if not ((self.number_of_samples_per_dist + envs) <= self.min_samples_per_dist).any() and it > self.min_it:
        if (torch.sum(self.number_of_samples_per_dist + envs)/self.env.num_envs > self.min_samples_per_dist) and it > self.min_it:
            self.geom_opt_step(it)
        else:
            self.number_of_samples_per_dist += envs

            # Sample a geometry from each distribution in envs
            for i in range(self.env.num_envs):
                if envs[i]:
                    self.geometry[i, self.geom_mask == 1] = self.distribution_log[i].sample()

            # Clip the geometry to the valid range
            self.geometry = torch.clamp(self.geometry, 0, 1)

    def update_distributions(self):
        # Update the mean, and L_cholseky
        self.mean += self.mean_gradient * self.mean_learning_rate
        # self.L_cholseky += self.L_cholseky_grad * self.mean_learning_rate

        # TODO: ensure that the resulting distribution is positive definite

    def estimate_gradient(self):
        # scale the reward
        factor = (self.min_samples_per_dist / self.number_of_samples_per_dist)
        self.rewards *= factor

        self.mean_gradient = torch.zeros(self.mean.size(), device=self.device)
        self.L_cholseky_grad = torch.zeros(self.L_cholseky.size(), device=self.device)
        # self.variances_gradient = torch.zeros(self.variances.size(), device=self.device)
        # self.covariances_gradient = torch.zeros(self.covariances.size(), device=self.device)

        for i, reward in enumerate(self.rewards):
            self.mean_gradient += reward * self.mean_noise[i]
            self.L_cholseky_grad += reward * self.L_cholseky_noise[i]
        
        self.mean_gradient /= self.rewards.numel() 
        self.L_cholseky_grad /= self.rewards.numel()
        print("mean_gradient", self.mean_gradient)

        # clip the gradients to be half of the std of the noise
        self.mean_gradient = torch.clamp(self.mean_gradient, -0.5 * self.mean_std, 0.5 * self.mean_std)
        # TODO find way to clip the cholseky gradient

        # estimate the gradient for the variances and covariances
        self.variances_gradient = torch.diag(self.L_cholseky_grad)
        lower_triangular_indices = torch.tril_indices(self.num_joints, self.num_joints, offset=-1)
        self.covariances_gradient = self.L_cholseky_grad[lower_triangular_indices[0], lower_triangular_indices[1]]



    def store_reward(self, reward, it):
        if it > self.min_it:
            if self.first_log:
                self.number_of_samples_per_dist = torch.ones(self.env.num_envs, device=self.device)
                self.first_log = False
                print("first log")
            self.rewards += reward

    def update_geom(self, it, envs = None):
        self.sample_geometry(envs, it)

    def geom_opt_step(self, it):
        self.estimate_gradient()
        self.update_distributions()
        self.initialize_distribution_log()
        # reset the loggs and rewards
        self.rewards = torch.zeros(self.env.num_envs, device=self.device)
        self.geometry_log = torch.tensor([], device=self.device)
        self.number_of_samples_per_dist = torch.zeros(self.env.num_envs, device=self.device)
        # reset all envs here?
        print("update the distributions")
        self.env.distribution_update(self.distribution_log)
        # self.env.reset() # maybe put back in, but for men episode lentht observation currently commented out


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


