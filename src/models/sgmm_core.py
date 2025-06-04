import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.cluster import KMeans

class SemiSupervisedGMM:
    def __init__(self, n_components, n_classes, max_iter=200, tol=1e-3, device='cuda'):
        self.n_components = n_components
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.alpha = None  # Mixing coefficients
        self.mu = None  # Means
        self.sigma = None  # Covariance matrices
        self.beta = None  # Class-component probabilities

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        """
        Fits the semi-supervised GMM to the data.

        Args:
            X_labeled: Labeled data (numpy array).
            y_labeled: Labels for labeled data (numpy array).
            X_unlabeled: Unlabeled data (numpy array).
        """
        print("-" * 30)
        print("Starting training of Semi-Supervised GMM...")
        print(f"Labeled samples: {X_labeled.shape[0]}, Unlabeled samples: {X_unlabeled.shape[0]}")
        print(f"Device: {self.device}")

        # Convert data to PyTorch tensors
        X_labeled = torch.from_numpy(X_labeled).to(dtype=torch.float64, device=self.device)
        y_labeled = torch.from_numpy(y_labeled).long().to(self.device)
        X_unlabeled = torch.from_numpy(X_unlabeled).to(dtype=torch.float64, device=self.device)

        X_all = torch.cat([X_labeled, X_unlabeled])

        # Initialize parameters using KMeans
        self._initialize_parameters(X_all)

        log_likelihood_old = -np.inf

        for iteration in range(self.max_iter):
            # E-step: Calculate responsibilities
            gamma_labeled, gamma_unlabeled = self._e_step(X_labeled, y_labeled, X_unlabeled)

            # M-step: Update parameters
            self._m_step(X_labeled, y_labeled, X_unlabeled, gamma_labeled, gamma_unlabeled)

            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X_labeled, y_labeled, X_unlabeled)
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1} complete, Log-Likelihood: {log_likelihood.item():.4f}")

            # Check for convergence
            if torch.abs(log_likelihood - log_likelihood_old) < self.tol:
                print(f"\nModel converged at iteration {iteration+1}")
                break
            log_likelihood_old = log_likelihood

        print("Model training complete!")

    def _initialize_parameters(self, X):
        """
        Initializes the parameters of the GMM.

        Args:
            X: Combined labeled and unlabeled data (PyTorch tensor).
        """
        kmeans = KMeans(
            n_clusters=self.n_components,
            init='k-means++',
            n_init=20,
            random_state=42
        ).fit(X.cpu().numpy())
        assignments = kmeans.labels_
        counts = np.bincount(assignments)

        self.alpha = torch.from_numpy(counts / counts.sum()).to(device=self.device, dtype=torch.float64)
        self.mu = torch.from_numpy(kmeans.cluster_centers_).to(device=self.device, dtype=torch.float64)

        covariances = []
        for i in range(self.n_components):
            data_i = X.cpu().numpy()[assignments == i]
            if len(data_i) > 1:
                cov_i = np.cov(data_i, rowvar=False)
            else:
                cov_i = np.eye(X.shape[1])
            covariances.append(cov_i)
        self.sigma = torch.from_numpy(np.stack(covariances, axis=0)).to(device=self.device, dtype=torch.float64)
        self.init_sigma = self.sigma.clone()  # Save initial covariance matrices

        self.beta = torch.rand(self.n_classes, self.n_components, device=self.device, dtype=torch.float64)
        self.beta /= self.beta.sum(dim=0, keepdim=True)  # Normalize

        print(f"  alpha shape: {self.alpha.shape}")
        print(f"  mu shape: {self.mu.shape}")
        print(f"  sigma shape: {self.sigma.shape}")
        print(f"  beta shape: {self.beta.shape}")
        print("Parameter initialization complete!")

    def _e_step(self, X_labeled, y_labeled, X_unlabeled):
        """
        Performs the E-step of the EM algorithm.

        Args:
            X_labeled: Labeled data (PyTorch tensor).
            y_labeled: Labels for labeled data (PyTorch tensor).
            X_unlabeled: Unlabeled data (PyTorch tensor).

        Returns:
            gamma_labeled: Responsibilities for labeled data.
            gamma_unlabeled: Responsibilities for unlabeled data.
        """
        # Calculate responsibilities for labeled data
        log_probs_labeled = torch.zeros((X_labeled.shape[0], self.n_components), device=self.device, dtype=torch.float64)
        for l in range(self.n_components):
            mvn = MultivariateNormal(self.mu[l], self.sigma[l])
            log_probs_labeled[:, l] = torch.log(self.alpha[l]) + mvn.log_prob(X_labeled) + torch.log(self.beta[y_labeled, l])

        log_sum_exp_labeled = torch.logsumexp(log_probs_labeled, dim=1, keepdim=True)
        gamma_labeled = torch.exp(log_probs_labeled - log_sum_exp_labeled)

        # Calculate responsibilities for unlabeled data
        log_probs_unlabeled = torch.zeros((X_unlabeled.shape[0], self.n_components), device=self.device, dtype=torch.float64)
        for l in range(self.n_components):
            mvn = MultivariateNormal(self.mu[l], self.sigma[l])
            log_probs_unlabeled[:, l] = torch.log(self.alpha[l]) + mvn.log_prob(X_unlabeled)

        log_sum_exp_unlabeled = torch.logsumexp(log_probs_unlabeled, dim=1, keepdim=True)
        gamma_unlabeled = torch.exp(log_probs_unlabeled - log_sum_exp_unlabeled)

        return gamma_labeled, gamma_unlabeled

    def _m_step(self, X_labeled, y_labeled, X_unlabeled, gamma_labeled, gamma_unlabeled):
        """
        Performs the M-step of the EM algorithm.

        Args:
            X_labeled: Labeled data (PyTorch tensor).
            y_labeled: Labels for labeled data (PyTorch tensor).
            X_unlabeled: Unlabeled data (PyTorch tensor).
            gamma_labeled: Responsibilities for labeled data.
            gamma_unlabeled: Responsibilities for unlabeled data.
        """
        resp_total = gamma_labeled.sum(dim=0) + gamma_unlabeled.sum(dim=0)

        # Update mu
        for l in range(self.n_components):
            numerator = (gamma_labeled[:, l, None] * X_labeled).sum(dim=0) + (gamma_unlabeled[:, l, None] * X_unlabeled).sum(dim=0)
            denominator = gamma_labeled[:, l].sum() + gamma_unlabeled[:, l].sum()
            self.mu[l] = numerator / denominator

        # Update sigma
        for l in range(self.n_components):
            diff_labeled = X_labeled - self.mu[l]
            sigma_labeled = (gamma_labeled[:, l, None, None] * torch.matmul(diff_labeled[:, :, None], diff_labeled[:, None, :])).sum(dim=0)

            diff_unlabeled = X_unlabeled - self.mu[l]
            sigma_unlabeled = (gamma_unlabeled[:, l, None, None] * torch.matmul(diff_unlabeled[:, :, None], diff_unlabeled[:, None, :])).sum(dim=0)

            denominator = gamma_labeled[:, l].sum() + gamma_unlabeled[:, l].sum()
            self.sigma[l] = (sigma_labeled + sigma_unlabeled) / denominator

            # Ensure that no covariance is too small
            s = torch.linalg.svdvals(self.sigma[l])
            if torch.min(s) < np.finfo(float).eps:
                # Reset to initial value if covariance matrix is too small
                self.sigma[l] = self.init_sigma[l].clone()
                print(f"Component {l} covariance matrix reset to initial value due to small singular values.")

        # Update alpha
        self.alpha = (gamma_labeled.sum(dim=0) + gamma_unlabeled.sum(dim=0)) / resp_total

        # Update beta(EM-I)
        for k in range(self.n_classes):
            numerator = gamma_labeled[y_labeled == k].sum(dim=0)
            denominator = gamma_labeled.sum(dim=0)
            self.beta[k] = numerator / denominator

        '''
        # Update beta (EM-II)
        for k in range(self.n_classes):
            for l in range(self.n_components):
                # Numerator for labeled data
                numerator_labeled = gamma_labeled[y_labeled == k, l].sum()

                # Numerator for unlabeled data: P(l, c_i=k|x_i, theta^t) = self.beta[k, l] * gamma_unlabeled[:, l]
                numerator_unlabeled = (self.beta[k, l] * gamma_unlabeled[:, l]).sum()

                # Denominator for labeled data
                denominator_labeled = gamma_labeled[:, l].sum()

                # Denominator for unlabeled data
                denominator_unlabeled = gamma_unlabeled[:, l].sum()

                self.beta[k, l] = (numerator_labeled + numerator_unlabeled) / (denominator_labeled + denominator_unlabeled)
        '''
        
    def _compute_log_likelihood(self, X_labeled, y_labeled, X_unlabeled):
        """
        Computes the log-likelihood of the data given the model parameters.

        Args:
            X_labeled: Labeled data (PyTorch tensor).
            y_labeled: Labels for labeled data (PyTorch tensor).
            X_unlabeled: Unlabeled data (PyTorch tensor).

        Returns:
            The log-likelihood of the data.
        """

        # Log-likelihood for labeled data
        labeled_log_probs = []
        for l in range(self.n_components):
            mvn = MultivariateNormal(self.mu[l], self.sigma[l])
            log_p = torch.log(self.alpha[l]) + mvn.log_prob(X_labeled) + torch.log(self.beta[y_labeled, l])
            labeled_log_probs.append(log_p.unsqueeze(1))
        labeled_log_probs = torch.cat(labeled_log_probs, dim=1)
        labeled_ll = torch.logsumexp(labeled_log_probs, dim=1).sum()

        # Log-likelihood for unlabeled data
        unlabeled_log_probs = []
        for l in range(self.n_components):
            mvn = MultivariateNormal(self.mu[l], self.sigma[l])
            log_p = torch.log(self.alpha[l]) + mvn.log_prob(X_unlabeled)
            unlabeled_log_probs.append(log_p.unsqueeze(1))
        unlabeled_log_probs = torch.cat(unlabeled_log_probs, dim=1)
        unlabeled_ll = torch.logsumexp(unlabeled_log_probs, dim=1).sum()

        return labeled_ll + unlabeled_ll

    def predict(self, X, return_probs=False):
        """
        Predicts class labels and confidence probabilities for the input data.

        Args:
            X (np.ndarray): Input data matrix of shape (n_samples, n_features)
            return_probs (bool): If True, returns full probability matrix
        
        Returns:
            tuple: 
                - pred_labels (np.ndarray): Predicted class labels, shape (n_samples,)
                - confidences (np.ndarray): Confidence probabilities, 
                    shape (n_samples,) when return_probs=False
                    shape (n_samples, n_classes) when return_probs=True
        """
        print(f"\nPredicting {X.shape[0]} samples...")
        
        # Convert input to tensor
        X_tensor = torch.from_numpy(X).to(
            dtype=torch.float64, 
            device=self.device
        )
        
        # Calculate component responsibilities (E-step)
        log_probs = torch.zeros(
            (X_tensor.shape[0], self.n_components),
            device=self.device,
            dtype=torch.float64
        )
        
        # Vectorized computation for efficiency
        for l in range(self.n_components):
            mvn = MultivariateNormal(self.mu[l], self.sigma[l])
            log_probs[:, l] = torch.log(self.alpha[l]) + mvn.log_prob(X_tensor)
        
        # Normalize responsibilities
        gamma = torch.softmax(log_probs, dim=1)
        
        # Calculate class probabilities: P(y=k|x) = sum_l [gamma(x)_l * beta_k^l]
        class_probs = torch.mm(gamma, self.beta.T)  # shape (n_samples, n_classes)
        
        # Get predictions and confidence scores
        pred_labels = torch.argmax(class_probs, dim=1).cpu().numpy()
        confidences = torch.max(class_probs, dim=1)[0].cpu().numpy()
        
        # Full probability matrix if requested
        if return_probs:
            confidences = class_probs.cpu().numpy()
        
        print(f"Prediction completed.")
        
        return pred_labels, confidences