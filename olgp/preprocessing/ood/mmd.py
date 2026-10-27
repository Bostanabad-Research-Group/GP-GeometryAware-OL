import torch
import os
import matplotlib.pyplot as plt
import hydra
import logging
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from olgp.utils import set_seed
from olgp.gpp.config import configure_logger, logger
from pathlib import Path

class MMDComputer:
    def __init__(self, pc_train):
        """
        Args:
            lengthscale (float): The 'l' parameter for the Gaussian kernel.
        """
        # Calculate the median heuristic dynamically based on a sample of data
        sample_pc = pc_train[:10].reshape(-1, pc_train.shape[-1]) # Flatten to [N*P, d]
        # Compute pairwise distances
        with torch.no_grad():
            # Use a subset of points to avoid massive memory blowup
            g = torch.Generator().manual_seed(0)
            idx = torch.randperm(sample_pc.shape[0], generator=g)[:5000]
            subset = sample_pc[idx]

            d_squared = torch.cdist(subset, subset) ** 2
            # Get the median of the non-zero distances
            median_dist = torch.median(torch.sqrt(d_squared[d_squared > 0]))
            print(f"The Median Heuristic suggests a lengthscale of: {median_dist.item():.4f}")

        lengthscale = median_dist.item()
        lengthscale = round(lengthscale, 1)
        self.gamma = 1.0 / (2 * (lengthscale ** 2))


    def compute_self_terms(self, X, batch_size):
        """
        Computes the first (or second) term of the MMD equation: sum(K(X, X)) / P^2
        X: (N, P, d)
        Returns: (N,) tensor of self-MMD values
        """
        N, P, _ = X.shape
        self_terms = torch.empty(N, device=X.device, dtype=X.dtype)

        for i in range(0, N, batch_size):
            batch_X = X[i : i + batch_size]
            d_squared = torch.cdist(batch_X, batch_X) ** 2
            K_XX = torch.exp(-self.gamma * d_squared)
            self_terms[i : i + batch_size] = K_XX.sum(dim=(1, 2)) / (P ** 2)

        return self_terms

    def compute_full_mmd_matrix(self, X, Y, batch_size_x=10, batch_size_y=10):
        """
        Computes the full N x M MMD matrix between X and Y.
        X: (N, P, d)
        Y: (M, P, d)
        """
        print(X.shape)
        N, P, _ = X.shape
        M = Y.shape[0]
        device = X.device

        print("Precomputing self-similarity terms...")
        term_XX = self.compute_self_terms(X, batch_size_x)
        term_YY = self.compute_self_terms(Y, batch_size_y)

        mmd_matrix = torch.empty((N, M), device=device, dtype=X.dtype)

        print(f"Computing cross-similarity terms ({N}x{M})...")
        for i in tqdm(range(0, N, batch_size_x)):
            x_chunk = X[i : i + batch_size_x]

            for j in range(0, M, batch_size_y):
                y_chunk = Y[j : j + batch_size_y]

                d_squared = torch.cdist(x_chunk.unsqueeze(1), y_chunk.unsqueeze(0)) ** 2
                K_XY = torch.exp(-self.gamma * d_squared)
                term_XY = K_XY.sum(dim=(2, 3)) / (P ** 2)

                mmd_block = (term_XX[i : i + batch_size_x].unsqueeze(1) +
                             term_YY[j : j + batch_size_y].unsqueeze(0) -
                             2 * term_XY)

                mmd_matrix[i : i + batch_size_x, j : j + batch_size_y] = mmd_block

        return mmd_matrix

@hydra.main(version_base=None, config_path="../../../configs", config_name="config_transolver")
def main(cfg: DictConfig):
    problem     = cfg.problem.name
    random_seed = cfg.settings.random_seed
    set_seed(random_seed)

    configure_logger(level=logging.DEBUG)
    logger.info(OmegaConf.to_yaml(cfg))

    pre_path_test = os.path.join(cfg.workdir.base_path, cfg.ablation.preprocess_subdir_test)
    Path(pre_path_test).mkdir(parents=True, exist_ok=True)

    data = torch.load(os.path.join(pre_path_test, f"{cfg.problem.name}_data.pt"), weights_only=True)
    pointcloud = data["pointcloud"]
    output_features = data["output_features"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pc_train = pointcloud[:cfg.problem.data.N_train].to(device)
    pc_test = pointcloud[-cfg.problem.data.N_test:].to(device)
    out_test = output_features[-cfg.problem.data.N_test:].to(device)
    print(f"Train PC shape: {pc_train.shape}, Test PC shape: {pc_test.shape}")

    BATCH_SIZE = 1
    mmd_computer = MMDComputer(pc_train)
    # MMD between every train and test geometry
    mmd_matrix_train_test_gpu = mmd_computer.compute_full_mmd_matrix(
        pc_train, pc_test,
        batch_size_x=BATCH_SIZE,
        batch_size_y=BATCH_SIZE
    )

    # OOD score of a test geometry: its minimum MMD to any training geometry
    min_mmd_to_train, closest_train_idx = mmd_matrix_train_test_gpu.min(dim=0)

    # Most OOD first
    sorted_ood_scores, sorted_test_indices = torch.sort(min_mmd_to_train, descending=True)

    print("\n--- TEST GEOMETRIES ORDERED BY GEOMETRIC NOVELTY (Most OOD to Least) ---")
    for rank, (score, idx) in enumerate(zip(sorted_ood_scores, sorted_test_indices)):
        print(f"Rank {rank+1:3d} | Test Index {idx.item():3d} | Min MMD to Train Dist: {score.item():.6f}")

    # Plot the 5 most and 5 least OOD test geometries
    most_ood_indices = sorted_test_indices[:5].cpu().numpy()
    least_ood_indices = sorted_test_indices[-5:].cpu().numpy()

    pc_test_cpu = pc_test.cpu().numpy()
    out_test_cpu = out_test.cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("OOD Detection via MMD (Test Set vs Train Distribution)", fontsize=16)

    # Top row: most OOD
    for i, idx in enumerate(most_ood_indices):
        ax = axes[0, i]
        ax.scatter(pc_test_cpu[idx, :, 0], pc_test_cpu[idx, :, 1], s=2, c=out_test_cpu[idx], alpha=1.0)
        ax.set_title(f"Rank {i+1} MOST OOD\n(Idx: {idx})")
        ax.axis('equal')

    # Bottom row: least OOD
    for i, idx in enumerate(reversed(least_ood_indices)):
        ax = axes[1, i]
        ax.scatter(pc_test_cpu[idx, :, 0], pc_test_cpu[idx, :, 1], s=2, c=out_test_cpu[idx], alpha=1.0)
        ax.set_title(f"Rank {i+1} LEAST OOD\n(Idx: {idx})")
        ax.axis('equal')

    plt.tight_layout()
    plot_path = os.path.join(pre_path_test, f"{problem}_similarity_plot.png")
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")

    torch.save({
            "mmd_train_train": None,
            "mmd_train_test": mmd_matrix_train_test_gpu.cpu(),
            "mmd_test_test": None,
            "ood_scores": min_mmd_to_train.cpu(),
        }, os.path.join(pre_path_test, rf"{problem}_data_mmd.pt"))

    print(f"Saved MMD matrices and OOD scores to: {os.path.join(pre_path_test, rf'{problem}_data_mmd.pt')}")

if __name__ == '__main__':
    main()
