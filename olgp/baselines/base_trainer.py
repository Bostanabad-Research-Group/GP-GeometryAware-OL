from abc import ABC, abstractmethod
import os
import torch
import numpy as np
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

# Import shared utilities
from olgp.utils import set_seed, get_env_metadata
from olgp.metrics import compute_mean_rL2
from olgp.gpp.config import configure_logger, logger
from olgp.plot import plot_training_dynamics

class BaselineTrainer(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        configure_logger(level=logging.DEBUG)
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

        # State
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.history = []

    def setup(self):
        """Orchestrator for setup."""
        # Set seed
        set_seed(self.cfg.settings.random_seed)

        self.load_data()       # Child implements this
        self.build_model()     # Child implements this
        self.setup_optimizer() # Child implements this

    def train(self):
        """Main shared training loop."""
        self.logger.info("--- Starting Training ---")
        start_time = datetime.now()
        # Reset peak memory to track it
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        best_val_l2 = float('inf')
        patience = 0
        limit = self.cfg.problem.training.convergence_patience
        epochs = self.cfg.problem.training.N_epochs

        pbar = tqdm(range(epochs), desc="Training", disable=not self.cfg.settings.tqdm)
        for epoch in pbar:
            epoch_start_time = datetime.now()
            # --- Train Step ---
            self.model.train()
            train_metrics = self.run_epoch(self.train_loader, is_train=True)

            # --- Val Step ---
            self.model.eval()
            with torch.no_grad():
                val_metrics = self.run_epoch(self.val_loader, is_train=False)

            epoch_duration = (datetime.now() - epoch_start_time).total_seconds()
            # --- Record & Log ---
            epoch_stats = {
                "epoch": epoch,
                "time_s": epoch_duration,
                **{f"train_{k}": v.item() if isinstance(v, np.generic) else v for k, v in train_metrics.items()},
                **{f"val_{k}": v.item() if isinstance(v, np.generic) else v for k, v in val_metrics.items()},
            }
            self.history.append(epoch_stats)

            pbar.set_postfix({"loss": f"{train_metrics['loss']:.4f}", "val_l2": f"{val_metrics['rel_l2']:.4f}"})
            self.logger.info(f"Epoch {epoch+1} | Train Loss: {train_metrics['loss']:.4f} | Val L2: {val_metrics['rel_l2']:.4f}")

            # --- Checkpointing ---
            if val_metrics['rel_l2'] < best_val_l2:
                best_val_l2 = val_metrics['rel_l2']
                patience = 0
                self._save_checkpoint("best_model.pth")
                self.logger.info(" | (*) New Best! Saved.")
            else:
                patience += 1

            if patience >= limit:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        self._finalize(start_time)

    def run_epoch(self, loader, is_train):
        """Generic epoch runner."""
        total_loss = 0
        total_l2 = 0

        for batch in loader:
            if is_train:
                self.optimizer.zero_grad()

            # This method differs per model, so it is abstract
            if is_train:
                loss, pred_unnorm, target_unnorm = self.compute_step(batch, is_train)
            elif not is_train:
                with torch.no_grad():
                    loss, pred_unnorm, target_unnorm = self.compute_step(batch, is_train)

            # Track metrics
            total_loss += loss
            # Calculate L2 (assuming compute_step must return unnormalized preds and targets)
            batch_l2 = compute_mean_rL2(target_unnorm, pred_unnorm)
            total_l2 += batch_l2

        return {
            "loss": total_loss / len(loader),
            "rel_l2": total_l2 / len(loader)
        }

    def _finalize(self, start_time):
        """Save benchmarks and plots."""
        # Time
        duration = (datetime.now() - start_time).total_seconds()
        avg_epoch_time = sum(h['time_s'] for h in self.history) / len(self.history) if self.history else 0
        # Memory
        max_memory = 0
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        # Total Parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        total_params_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Save Benchmarks YAML
        benchmarks = {
            "env": get_env_metadata(self.cfg),
            "metrics": {
                "total_time_min": duration / 60,
                "avg_epoch_time_s": avg_epoch_time,
                "best_val_l2": min(h['val_rel_l2'] for h in self.history),
                "peak_memory_mb": max_memory,
                "total_params": total_params,
                "trainable_params": total_params_trainable
            },
            "history": self.history
        }
        with open(os.path.join(self.output_dir, "Benchmarks/benchmarks_train.yaml"), 'w') as f:
            yaml.dump(benchmarks, f)

        # Generate Plot (Shared logic)
        self._plot_dynamics()

    def _plot_dynamics(self):
        """Standardized Plotting for all models using shared utils."""
        # 1. Extract data
        epochs_arr = [entry['epoch'] for entry in self.history]
        train_loss = [entry['train_loss'] for entry in self.history]
        val_loss   = [entry['val_loss'] for entry in self.history]
        val_l2     = [entry['val_rel_l2'] for entry in self.history]

        # 2. Define path
        plot_path = os.path.join(self.output_dir, "Plots", "training_dynamics.png")

        # 3. Call shared function
        # Note: Since epochs are the same for all in this class, pass 'epochs_arr' for all epoch args
        plot_training_dynamics(
            train_loss=train_loss, train_epochs=epochs_arr,
            val_loss=val_loss,     val_epochs=epochs_arr,
            metric_values=val_l2,  metric_epochs=epochs_arr,
            save_path=plot_path,
            title="Training Dynamics",
            metric_label="Val Rel Error",
            logger=self.logger
        )

    def _save_checkpoint(self, filename):
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "X_mean": self.scaler_X.mean,
            "X_std": self.scaler_X.std,
            "U_mean": self.scaler_U.mean,
            "U_std": self.scaler_U.std,
        }
        torch.save(save_dict, os.path.join(self.output_dir, f"Models/{filename}"))

    def run(self):
        """Full pipeline execution."""
        self.logger.info(f"{'#' * 50}")
        self.logger.info(OmegaConf.to_yaml(self.cfg))
        start_time = datetime.now()

        self.setup()
        self.train()

        elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
        self.logger.info(f"[Time] Total Elapsed: {elapsed_minutes:.2f} minutes")
        self.logger.info(f"{'#' * 50}")

    def load_data(self):
        """
        Must setup:
            - self.train_loader
            - self.val_loader
            - self.scaler_X
            - self.scaler_U
        """
        self.train_loader = None
        self.val_loader = None
        self.scaler_X = None
        self.scaler_U = None

        # Call the subclass implementation
        self._load_data_impl()

        if self.train_loader is None or self.val_loader is None:
            raise ValueError("train_loader and val_loader must be set in _load_data_impl")
        if self.scaler_X is None or self.scaler_U is None:
            raise ValueError("scaler_X and scaler_U must be set in _load_data_impl")

    def build_model(self):
        self.model = None

        # Call the subclass implementation
        self._build_model_impl()

        if self.model is None:
            raise ValueError("model must be set in _build_model_impl")

    def setup_optimizer(self):
        self.optimizer = None
        self.scheduler = None

        # Call the subclass implementation
        self._setup_optimizer_impl()

        if self.optimizer is None or self.scheduler is None:
            raise ValueError("optimizer and scheduler must be set in _setup_optimizer_impl")

    # --- Abstract Methods ---
    @abstractmethod
    def _load_data_impl(self): raise NotImplementedError
    @abstractmethod
    def _build_model_impl(self): raise NotImplementedError
    @abstractmethod
    def _setup_optimizer_impl(self): raise NotImplementedError
    @abstractmethod
    def compute_step(self, batch, is_train: bool):
        """Forward pass, must define loss and return predictions and targets (unnormalized)."""
        raise NotImplementedError
