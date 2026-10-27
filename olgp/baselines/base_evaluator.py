from abc import ABC, abstractmethod
import os
import torch
import yaml
import logging
import time
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

# Import shared utilities
from olgp.utils import set_seed, get_env_metadata
from olgp.metrics import compute_mean_rL2
from olgp.gpp.config import configure_logger, logger

class BaselineEvaluator(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        configure_logger(level=logging.INFO)
        self.logger = logger
        self.tkwargs = {
            "dtype": getattr(torch, cfg.settings.dtype),
            "device": torch.device(cfg.settings.device),
        }
        self.device = self.tkwargs["device"]
        self.dtype = self.tkwargs["dtype"]

        # Setup Directories
        self.output_dir = HydraConfig.get().runtime.output_dir # Hydra sets this
        Path(self.output_dir + "/Benchmarks").mkdir(exist_ok=True, parents=True)
        Path(self.output_dir + "/Plots").mkdir(exist_ok=True, parents=True)
        Path(self.output_dir + "/Models").mkdir(exist_ok=True, parents=True)
        self.pre_path = os.path.join(cfg.workdir.base_path, cfg.ablation.preprocess_subdir_train)
        Path(self.pre_path).mkdir(parents=True, exist_ok=True)
        self.pre_path_test = os.path.join(cfg.workdir.base_path, cfg.ablation.preprocess_subdir_test)
        Path(self.pre_path_test).mkdir(parents=True, exist_ok=True)

        # State
        self.model = None
        self.test_loader = None

    def setup(self, set_fixed_seed=True):
        """Orchestrator: Loads model first (to get scaler stats), then data."""
        if set_fixed_seed:
            set_seed(self.cfg.settings.random_seed)

        self.load_model() # Child implements impl
        self.load_data()  # Child implements impl

    def evaluate(self):
        """Main evaluation loop."""
        self.logger.info("--- Starting Evaluation ---")

        # Start metrics
        start_time = datetime.now()
        # Reset peak memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # 1. Inference & Timing
        metrics, predictions, targets, inputs, extras = self._run_inference_loop()

        # 2. Save Metrics
        self._save_benchmarks(metrics, start_time)

        # 3. Generate Plots
        self._plot_results(inputs, targets, predictions, extras)

        self.logger.info(f"Evaluation Complete. Results saved to {self.output_dir}")

    def _run_inference_loop(self):
        self.model.eval()
        total_time = 0.0
        total_samples = 0

        all_preds = []
        all_targets = []
        all_input_coords = []
        all_extras = []

        # Warm-up batches, so GPU timings are not skewed
        self.logger.info("Warming up GPU...")
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if i > 5:
                    break
                self.compute_inference(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # --- Main Loop ---
        self.logger.info("Running Inference...")
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Start Timer
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()

                # Forward pass (child implements this)
                # Must return unnormalized predictions and targets
                # pred_un: [B, Npoints], target_un: : [B, Npoints], input_coords: [B, Npoints, 2], extras: {}
                pred_un, target_un, input_coords, extras = self.compute_inference(batch)
                # Stop Timer
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.time() - t0

                # Accumulate
                batch_size = target_un.shape[0]
                total_time += dt
                total_samples += batch_size

                # Store for aggregate metrics
                # Move to CPU to prevent OOM
                all_preds.append(pred_un.cpu())
                all_targets.append(target_un.cpu())
                if input_coords is not None:
                    all_input_coords.append(input_coords.cpu())
                if extras:
                    cpu_extras = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                  for k, v in extras.items()}
                    all_extras.append(cpu_extras)

        # Concatenate
        preds_tensor = torch.cat(all_preds, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        input_coords_tensor = torch.cat(all_input_coords, dim=0) if all_input_coords else None
        # Merge list of dicts into one dict of concatenated tensors
        # Example: [{x: T1}, {x: T2}] -> {x: cat(T1, T2)}
        merged_extras = {}
        if all_extras:
            keys = all_extras[0].keys()
            for k in keys:
                # Only concatenate if they are tensors
                first_val = all_extras[0][k]
                if isinstance(first_val, torch.Tensor):
                    merged_extras[k] = torch.cat([d[k] for d in all_extras], dim=0)
                else:
                    # For non-tensors (lists, strings), just keep the list
                    merged_extras[k] = [d[k] for d in all_extras]

        # --- Compute Metrics ---
        # 1. Relative L2
        rel_l2 = compute_mean_rL2(targets_tensor, preds_tensor)
        # 2. MSE / MAE
        mse = torch.mean((preds_tensor - targets_tensor)**2).item()
        mae = torch.mean(torch.abs(preds_tensor - targets_tensor)).item()

        # 3. Inference Speed
        avg_latency_ms = (total_time / total_samples) * 1000

        metrics = {
            "rel_l2": rel_l2.item(),
            "mse": mse,
            "mae": mae,
            "inference_latency_ms": avg_latency_ms,
            "samples_per_sec": total_samples / total_time,
        }

        return metrics, preds_tensor, targets_tensor, input_coords_tensor, merged_extras

    def _save_benchmarks(self, metrics, start_time):
        """Saves metrics to YAML."""
        # Time
        duration = (datetime.now() - start_time).total_seconds()
        # Memory
        max_memory = 0
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

        benchmarks = {
            "env": get_env_metadata(self.cfg),
            "metrics": metrics,
            "total_eval_time_min": duration / 60,
            "peak_memory_mb": max_memory,
        }
        filename = "benchmarks_eval.yaml" if self.cfg.problem.inference.split == "test" else "benchmarks_eval_train.yaml"
        out_path = os.path.join(self.output_dir, "Benchmarks", filename)
        with open(out_path, 'w') as f:
            yaml.dump(benchmarks, f)
        self.logger.info(f"Benchmarks saved: L2={metrics['rel_l2']:.4f} | Latency={metrics['inference_latency_ms']:.2f}ms")

    def _plot_results(self, input_coords, targets, preds, extras):
        """
        Generates comparison plots for best, median, and worst samples based on L2 error.
        Assumes 2D data (N_samples, N_points).
        """
        try:
            # Calculate error per sample to find best/worst
            # Flatten to [N, -1] for norm calculation
            diff = preds - targets
            sample_norms = torch.norm(diff.reshape(diff.shape[0], -1), p=2, dim=1)
            target_norms = torch.norm(targets.reshape(targets.shape[0], -1), p=2, dim=1)
            rel_errs = sample_norms / (target_norms + 1e-6)

            indices = torch.argsort(rel_errs)
            best_idx = indices[0]
            median_idx = indices[len(indices)//2]
            worst_idx = indices[-1]

            selection = {
                "best": (best_idx, rel_errs[best_idx]),
                "median": (median_idx, rel_errs[median_idx]),
                "worst": (worst_idx, rel_errs[worst_idx])
            }

            for name, (idx, err) in selection.items():
                self._plot_sample(name, idx, input_coords, targets, preds, err, extras)

        except Exception as e:
            self.logger.error(f"Plotting failed: {e}")

    def _plot_sample(self, name, idx, input_coords, targets, preds, err, extras):
        """
        Visualizes a single sample using Scatter plots (Mesh/Pointcloud style).
        """

        # Extract single sample data
        coords = input_coords[idx]  # [Points, 2]
        target = targets[idx]       # [Points]
        pred = preds[idx]           # [Points]

        # Determine number of subplots
        num_plots = 3
        fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
        # Common scatter args
        s = 20 # point size
        cmap = 'viridis'
        curr_ax = 0
        # Ground Truth
        sc1 = axes[curr_ax].scatter(coords[:,0], coords[:,1], c=target, s=s, cmap=cmap)
        axes[curr_ax].set_title("Target u(x)")
        plt.colorbar(sc1, ax=axes[curr_ax])
        # Prediction
        sc2 = axes[curr_ax+1].scatter(coords[:,0], coords[:,1], c=pred, s=s, cmap=cmap)
        axes[curr_ax+1].set_title(f"Pred (RelL2: {err:.4f})")
        plt.colorbar(sc2, ax=axes[curr_ax+1])
        # Absolute Error
        error_map = torch.abs(target - pred)
        sc3 = axes[curr_ax+2].scatter(coords[:,0], coords[:,1], c=error_map, s=s, cmap='inferno')
        axes[curr_ax+2].set_title("Abs Error |u - u_pred|")
        plt.colorbar(sc3, ax=axes[curr_ax+2])
        # Formatting
        for ax in axes:
            ax.set_aspect('equal')
            ax.axis('off')
        # Save
        save_name = f"{name}_sample.png" if self.cfg.problem.inference.split == "test" else f"{name}_train_sample.png"
        save_path = os.path.join(self.output_dir, "Plots", save_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def load_model(self):
        self.model = None
        self.scaler_X = None
        self.scaler_U = None

        # Child loads the architecture
        self._load_model_impl()

        if self.model is None:
            raise ValueError("Model not initialized")
        if self.scaler_X is None or self.scaler_U is None:
            raise ValueError("scaler_X and scaler_U must be set in _load_data_impl")

    def load_data(self):
        self.test_loader = None

        self._load_data_impl()

        if self.test_loader is None:
            raise ValueError("Test loader not initialized")

    def run(self):
        """Full pipeline execution."""
        self.logger.info(f"{'#' * 50}")
        self.logger.info(OmegaConf.to_yaml(self.cfg))
        start_time = datetime.now()

        self.setup()
        self.evaluate()

        elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
        self.logger.info(f"[Time] Total Elapsed: {elapsed_minutes:.2f} minutes")
        self.logger.info(f"{'#' * 50}")

    # --- Abstracts ---
    @abstractmethod
    def _load_model_impl(self): raise NotImplementedError
    @abstractmethod
    def _load_data_impl(self): raise NotImplementedError
    @abstractmethod
    def compute_inference(self, batch): raise NotImplementedError
