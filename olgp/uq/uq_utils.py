import torch
import numpy as np

def compute_probabilistic_metrics(mu, sigma, y):
    """Computes the relative L2 error, NLL, CRPS, ACE and interval score."""
    mu, sigma, y = mu.cpu(), sigma.cpu(), y.cpu()

    # Relative L2 error, averaged over samples
    mu_batch = mu.view(mu.shape[0], -1)
    y_batch = y.view(y.shape[0], -1)

    l2_error_per_sample = torch.linalg.norm(y_batch - mu_batch, dim=1)
    l2_norm_y_per_sample = torch.linalg.norm(y_batch, dim=1)

    eps = 1e-6
    rel_l2_per_sample = l2_error_per_sample / torch.clamp(l2_norm_y_per_sample, min=eps)
    rel_l2_val = torch.mean(rel_l2_per_sample).item()

    # The other metrics are averaged over all points
    mu_flat = mu.flatten()
    sigma_flat = sigma.flatten()
    y_flat = y.flatten()

    sigma_flat = torch.clamp(sigma_flat, min=eps)
    variance_flat = sigma_flat ** 2

    # Gaussian NLL
    log_2pi = torch.log(torch.tensor(2.0 * np.pi))

    nll_log_term = torch.log(sigma_flat)
    nll_error_term = ((y_flat - mu_flat)**2) / variance_flat
    nll_val = torch.mean(0.5 * log_2pi + nll_log_term + 0.5 * nll_error_term).item()

    # Closed-form Gaussian CRPS
    z = (y_flat - mu_flat) / sigma_flat

    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * torch.exp(-0.5 * z**2)
    cdf = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
    crps_val = torch.mean(sigma_flat * (z * (2 * cdf - 1) + 2 * pdf - (1.0 / np.sqrt(np.pi)))).item()

    # Average calibration error over 101 quantile levels
    pit_values = 0.5 * (1.0 + torch.erf((y_flat - mu_flat) / (sigma_flat * np.sqrt(2.0))))
    alphas = torch.linspace(0, 1, steps=101)
    empirical_coverage = torch.mean((pit_values.unsqueeze(1) <= alphas.unsqueeze(0)).float(), dim=0)
    ace_val = torch.mean(torch.abs(empirical_coverage - alphas)).item()

    # Interval score of the 95% interval
    alpha = 0.05
    z_alpha = 1.96
    u = mu_flat + z_alpha * sigma_flat
    l = mu_flat - z_alpha * sigma_flat
    is_score = (u - l)
    mask_lower = y_flat < l
    is_score[mask_lower] += (2.0 / alpha) * (l[mask_lower] - y_flat[mask_lower])
    mask_upper = y_flat > u
    is_score[mask_upper] += (2.0 / alpha) * (y_flat[mask_upper] - u[mask_upper])
    nois_val = torch.mean(is_score).item()

    return rel_l2_val, nll_val, crps_val, ace_val, nois_val

def compute_energy_score_torch(samples, truth):
    """Energy score averaged over the test samples.

    samples: [M, N_test, ...] predictive samples; truth: [N_test, ...].
    """
    M = samples.shape[0]

    samples_flat = samples.view(M, samples.shape[1], -1)
    truth_flat = truth.view(truth.shape[0], -1)

    # E||X - y||
    term1 = torch.norm(samples_flat - truth_flat.unsqueeze(0), p=2, dim=-1).mean(dim=0)

    # 0.5 E||X - X'||, looped over pairs to avoid an [M, M, N_test, Points] tensor
    term2 = torch.zeros_like(term1)
    for i in range(M):
        for j in range(M):
            if i != j:
                term2 += torch.norm(samples_flat[i] - samples_flat[j], p=2, dim=-1)
    term2 = term2 / (2 * M * (M - 1))

    energy_score_per_item = term1 - term2
    return energy_score_per_item.mean().item()
