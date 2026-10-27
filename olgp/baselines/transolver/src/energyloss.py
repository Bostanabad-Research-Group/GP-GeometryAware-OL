import torch
import torch.nn as nn

class EnergyScoreLoss(nn.Module):
    def __init__(self, p=2, reduction='mean'):
        """
        Energy Score Loss for Probabilistic Neural Operators (PNO).

        Args:
            p (int): The norm degree (default: 2 for L2 norm / Hilbert space).
            reduction (str): 'mean', 'sum', or 'none' for batch reduction.
        """
        super(EnergyScoreLoss, self).__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): The M predicted samples.
                              Expected shape: (Batch_size, M_samples, *spatial_dims)
            y (torch.Tensor): The ground truth observation.
                              Expected shape: (Batch_size, *spatial_dims)
        Returns:
            torch.Tensor: The computed Energy Score.
        """
        B, M = x.shape[0], x.shape[1]

        # Flatten spatial dimensions so we can compute norms easily
        # x_flat shape: (B, M, D) where D is the flattened spatial dimension size
        x_flat = x.view(B, M, -1)

        # y_flat shape: (B, 1, D) - unsqueezed to broadcast against the M samples
        y_flat = y.view(B, -1).unsqueeze(1)

        # ------------------------------------------------------------------
        # Term 1: Accuracy Penalty (Mean distance from samples to target)
        # Formula: (1 / M) * sum( || x^{(m)} - y ||_p )
        # ------------------------------------------------------------------
        # Calculate norm along the flattened spatial dimension (dim=-1)
        dist_to_target = torch.norm(x_flat - y_flat, p=self.p, dim=-1)  # Shape: (B, M)
        term1 = dist_to_target.mean(dim=1)                              # Shape: (B,)

        # ------------------------------------------------------------------
        # Term 2: Diversity Reward (Pairwise distances between samples)
        # Formula: (1 / 2*M*(M-1)) * sum( || x^{(m)} - x^{(k)} ||_p )
        # ------------------------------------------------------------------
        if M > 1:
            # pairwise_dists shape: (B, M, M)
            pairwise_dists = torch.cdist(x_flat, x_flat, p=self.p)

            # Sum all pairwise distances and divide by the normalization factor
            term2 = pairwise_dists.sum(dim=(1, 2)) / (2 * M * (M - 1))  # Shape: (B,)
        else:
            # If M=1, there is no variance to measure, so the diversity term is 0
            term2 = 0.0

        # ------------------------------------------------------------------
        # Final Score: Accuracy - Diversity
        # ------------------------------------------------------------------
        energy_score = term1 - term2

        # Apply batch reduction
        if self.reduction == 'mean':
            return energy_score.mean()
        elif self.reduction == 'sum':
            return energy_score.sum()
        else:
            return energy_score
