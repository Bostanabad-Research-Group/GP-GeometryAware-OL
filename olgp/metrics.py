import torch
import numpy as np

def compute_mean_rL2(y_true, y_pred, eps=1e-8):
    """
    Computes the Mean Relative L2 error.
    Safe for both PyTorch tensors and NumPy arrays.
    """
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().squeeze().detach().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().squeeze().detach().numpy()

    diff_norm = np.linalg.norm(y_pred - y_true, axis = -1)
    # eps prevents division by zero if the ground truth set is zero
    true_norm = np.linalg.norm(y_true, axis = -1) + eps

    rl2 = np.mean(diff_norm / (true_norm ))
    return rl2
