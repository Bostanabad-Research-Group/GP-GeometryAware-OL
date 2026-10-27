import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from hydra.core.hydra_config import HydraConfig

import yaml
from olgp.uq.uq_utils import compute_probabilistic_metrics
from olgp.utils import get_env_metadata

class UQEvaluator():
    def __init__(self, cfg, pretrained_no_class, name, seeds=(0, 1, 2, 3, 4), fit=False):
        self.cfg = cfg
        self.pretrained_no_class = pretrained_no_class
        self.name = name
        self.seeds = seeds
        self.fit = cfg.problem.uq.fit
        self.base_deterministic_seed = 0
        self.evaluators = None
        self.X_test = None # [N_seeds, N_test, Npoints, 2 or 3] torch.Tensor
        self.Y_test = None
        self.pred_mean = None
        self.pred_std = None
        self.mmd_geom = None
        self.ood_quantiles = {} # filled only if MMD OOD scores are available

    def run(self):
        """Main method to run the evaluation."""
        self._load_base_evaluator()
        if self.evaluators is None:
            raise ValueError("Base evaluators not loaded. Ensure _load_base_evaluator() is implemented correctly.")

        # ----- Fit -----
        times, memories, fitted = self._fit()
        self.fitted = fitted
        fit_metrics = {
            "fit_time_avg_sec": np.mean(times).item() if times else 0.0,
            "fit_time_std_sec": np.std(times).item() if times else 0.0,
            "fit_performed": fitted,
            "fit_max_memory_mb": np.mean(memories).item() if memories else 0.0,
            "fit_max_memory_std_mb": np.std(memories).item() if memories else 0.0
        }
        # ----- Evaluate -----
        eval_times, eval_memories, energy_scores = self._evaluate()
        eval_metrics = {
            "eval_time_avg_sec": np.mean(eval_times).item() if eval_times else 0.0,
            "eval_time_std_sec": np.std(eval_times).item() if eval_times else 0.0,
            "eval_max_memory_mb": np.mean(eval_memories).item() if eval_memories else 0.0,
            "eval_max_memory_std_mb": np.std(eval_memories).item() if eval_memories else 0.0
        }

        if self.pred_mean is None or self.pred_std is None or self.X_test is None or self.Y_test is None:
            raise ValueError("Evaluation did not produce predictions or test data. Ensure _evaluate() is implemented correctly.")

        # Compute metrics
        metrics_storage = { 'rL2': {'values':[]}, 'nll': {'values':[]}, 'crps': {'values':[]}, 'ace': {'values':[]}, 'is': {'values':[]}, 'energy': {'values':[]} }
        for i in range(len(self.pred_mean)):
            print(f"Shapes for Seed {self.seeds[i]}: pred_mean={self.pred_mean[i].shape}, pred_std={self.pred_std[i].shape}, Y_test={self.Y_test[i].shape}")
            rL2, nll, crps, ace, nois = compute_probabilistic_metrics(self.pred_mean[i], self.pred_std[i], self.Y_test[i])
            metrics_storage['rL2']['values'].append(rL2)
            metrics_storage['nll']['values'].append(nll)
            metrics_storage['crps']['values'].append(crps)
            metrics_storage['ace']['values'].append(ace)
            metrics_storage['is']['values'].append(nois)
        metrics_storage['energy']['values'] = energy_scores
        for metric, values in metrics_storage.items():
            mean_val = np.mean(values['values'])
            std_val = np.std(values['values'])
            print(f"  > {metric.upper()}: {mean_val:.5f} ± {std_val:.5f}")
            metrics_storage[metric]['mean'] = mean_val.item()
            metrics_storage[metric]['std'] = std_val.item()

        # Metrics per geometry-novelty quartile (only if MMD OOD scores are available)
        print("\n" + "="*50)
        print("--- METRICS PER OOD ---")
        if "ood_scores" in self.mmd_geom[0]:
            # Extract scores and sort descending (Highest distance = Most OOD)
            ood_scores = self.mmd_geom[0]["ood_scores"].to(self.Y_test[0].device)
            sorted_scores, sorted_idx = torch.sort(ood_scores, descending=True)

            n_test = len(sorted_idx)
            q_size = n_test // 4

            # Define the 4 quantiles
            self.ood_quantiles = {
                "Q4": sorted_idx[0 : q_size],
                "Q3": sorted_idx[q_size : 2*q_size],
                "Q2": sorted_idx[2*q_size : 3*q_size],
                "Q1": sorted_idx[3*q_size : ]
            }

            metrics_storage['quantiles'] = {}

            for q_name, q_indices in self.ood_quantiles.items():
                avg_dist = ood_scores[q_indices].mean().item()
                print(f"\n>> {q_name} | N={len(q_indices)} | Avg MMD Dist={avg_dist:.6f}")

                # Temporary storage for seeds in this quantile
                q_mets = {m: [] for m in ['rL2', 'nll', 'crps', 'ace', 'is']}

                for i in range(len(self.pred_mean)): # Loop over seeds
                    # Slice tensors for only the geometries in this quantile
                    q_mean = self.pred_mean[i][q_indices]
                    q_std = self.pred_std[i][q_indices]
                    q_y = self.Y_test[i][q_indices]

                    rL2, nll, crps, ace, nois = compute_probabilistic_metrics(q_mean, q_std, q_y)

                    q_mets['rL2'].append(rL2)
                    q_mets['nll'].append(nll)
                    q_mets['crps'].append(crps)
                    q_mets['ace'].append(ace)
                    q_mets['is'].append(nois)

                # Print and save to main storage
                metrics_storage['quantiles'][q_name] = {}
                for metric in ['rL2', 'nll', 'crps', 'ace', 'is']:
                    m_mean = np.mean(q_mets[metric])
                    m_std = np.std(q_mets[metric])
                    print(f"    - {metric.upper():<4}: {m_mean:.5f} ± {m_std:.5f}")

                    # Store in the dictionary to be saved in the YAML file
                    metrics_storage['quantiles'][q_name][metric] = {
                        'mean': m_mean.item(),
                        'std': m_std.item()
                    }
        else:
            print("WARNING: 'ood_scores' not found in self.mmd_geom dictionary. Skipping quantile analysis.")

        print("="*50 + "\n")

        self._save_benchmarks(metrics_storage, fit_metrics, eval_metrics)

        # Plot
        self.plot(self.base_deterministic_seed)

    def _load_base_evaluator(self):
        self.evaluators = []
        self.mmd_geom = []
        for seed in self.seeds:
            tmp_cfg = copy.deepcopy(self.cfg)
            tmp_cfg.settings.random_seed = seed
            ev = self.pretrained_no_class(tmp_cfg)
            ev.output_dir = ev.output_dir.replace(f"seed_{self.cfg.settings.random_seed}", f"seed_{seed}")
            self.evaluators.append(ev)

            # MMD OOD scores are optional (generated by scripts/linux/transolver/run_mmd.sh)
            mmd_path = os.path.join(ev.pre_path_test, f"{self.cfg.problem.name}_data_mmd.pt")
            if os.path.exists(mmd_path):
                mmd_data = torch.load(mmd_path, weights_only=True)
            else:
                print(f"WARNING: MMD file not found at {mmd_path}. OOD quantile analysis will be skipped.")
                mmd_data = {}
            self.mmd_geom.append(mmd_data)

    def _fit(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _evaluate(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _save_benchmarks(self, metrics, fit_metrics, eval_metrics):
        """Saves metrics to YAML."""
        benchmarks = {
            "env": get_env_metadata(self.cfg),
            "fit_metrics": fit_metrics,
            "eval_metrics": eval_metrics,
            "metrics": metrics,
        }
        self.output_dir = HydraConfig.get().runtime.output_dir # Hydra sets this
        out_path = os.path.join(self.output_dir, "Benchmarks", f"benchmarks_eval_UQ_{self.name}.yaml")
        with open(out_path, 'w') as f:
            yaml.dump(benchmarks, f)
        print(f"Benchmarks saved: {out_path}")

        if self.fitted:
            train_cost = {
                "fit_metrics": fit_metrics,
                "eval_metrics": eval_metrics,
            }
            out_train_cost_path = os.path.join(self.output_dir, "Benchmarks", f"benchmarks_eval_UQ_{self.name}_train_cost.yaml")
            with open(out_train_cost_path, 'w') as f:
                yaml.dump(train_cost, f)
            print(f"Benchmarks train cost saved: {out_train_cost_path}")

    def plot(self, seed_idx=0):
        """Plots the test samples at the 25th, 50th and 75th percentiles of the base model error."""
        print("\n--- Identifying Baseline Quantiles from Base Model (Seed 0) ---")
        # 1. Load the base evaluator for seed 0
        base_eval = self.evaluators[self.base_deterministic_seed]
        base_model = base_eval.model
        base_model.eval()

        base_preds = []
        base_gts = []

        # 2. Run inference with the base model
        with torch.no_grad():
            for batch in base_eval.test_loader:
                if len(batch) == 2:
                    x, y_gt = batch
                else:
                    pos, fx, y_gt = batch
                    if pos.shape[-1] < self.cfg.problem.model.fun_dim + self.cfg.problem.model.d:
                        x = torch.cat([pos, fx], dim=-1)
                    else:
                        x = pos

                x = x.to(base_eval.device)

                y_nn_norm = base_model(x, None)

                # Reduce [B, N, N] or [B, N, 1] outputs to [B, N]
                if y_nn_norm.ndim == 3:
                    if y_nn_norm.shape[-1] == y_nn_norm.shape[1]:
                        y_nn_norm = y_nn_norm.mean(dim=-1)
                    elif y_nn_norm.shape[-1] == 1:
                        y_nn_norm = y_nn_norm.squeeze(-1)

                # Back to physical units
                pred_nn_real = base_eval.scaler_U.inverse(y_nn_norm.unsqueeze(-1)).squeeze(-1)
                target_real = base_eval.scaler_U.inverse(y_gt.to(base_eval.device)).squeeze(-1)

                base_preds.append(pred_nn_real.cpu())
                base_gts.append(target_real.cpu())

        base_preds = torch.cat(base_preds, dim=0)
        base_gts = torch.cat(base_gts, dim=0)

        # 3. Relative L2 error of the base model per test sample
        diff = base_preds - base_gts
        l2_diff = torch.norm(diff, p=2, dim=1)
        l2_true = torch.norm(base_gts, p=2, dim=1)

        rel_l2_per_sample = l2_diff / (l2_true + 1e-8)

        # 4. Find the quantile indices
        sorted_indices = torch.argsort(rel_l2_per_sample)
        n_samples = len(sorted_indices)

        q25_idx = sorted_indices[int(n_samples * 0.25)].item()
        q50_idx = sorted_indices[int(n_samples * 0.50)].item()
        q75_idx = sorted_indices[int(n_samples * 0.75)].item()

        print(f"Worst 5 Sample Indices: {sorted_indices[-5:].cpu().numpy()}")
        print(f"Best 5 Sample Indices: {sorted_indices[0:5].cpu().numpy()}")

        indices_to_plot = {
            'Base 25th Quantile': q25_idx,
            'Base 50th Quantile (Median)': q50_idx,
            'Base 75th Quantile': q75_idx
        }

        print(f"\nSelected Samples: {indices_to_plot}")

        # First, middle and last sample of each OOD quartile
        selected_ood_quantile_indices = {}
        for q_name, q_array in self.ood_quantiles.items():
            n = len(q_array)
            first_idx = q_array[0]
            middle_idx = q_array[n // 2]
            last_idx = q_array[-1]
            selected_ood_quantile_indices[q_name] = [first_idx, middle_idx, last_idx]
        for q_name, idx_list in selected_ood_quantile_indices.items():
            print(f"{q_name} Representative Indices: {idx_list}")

        # Store plot tensors
        data_to_store = {
            "all": {
                "pts": self.X_test.cpu().numpy(),
                "gt": self.Y_test.cpu().numpy(),
                "pred": self.pred_mean.cpu().numpy(),
                "std": self.pred_std.cpu().numpy(),
            },
            "global": {
                "indices": [q25_idx, q50_idx, q75_idx],
                "pts": self.X_test[:,[q25_idx, q50_idx, q75_idx]].cpu().numpy(),
                "gt": self.Y_test[:,[q25_idx, q50_idx, q75_idx]].cpu().numpy().flatten(),
                "pred": self.pred_mean[:,[q25_idx, q50_idx, q75_idx]].cpu().numpy().flatten(),
                "std": self.pred_std[:,[q25_idx, q50_idx, q75_idx]].cpu().numpy().flatten(),
            },
            "quantiles": {
                "q_indices": selected_ood_quantile_indices,
                # Data for each OOD quantile (Q1-Q4), empty if no MMD scores
                **{
                    q_name: {
                        "pts": self.X_test[:, q_idx].cpu().numpy(),
                        "gt": self.Y_test[:, q_idx].cpu().numpy(),
                        "pred": self.pred_mean[:, q_idx].cpu().numpy(),
                        "std": self.pred_std[:, q_idx].cpu().numpy(),
                    }
                    for q_name, q_idx in self.ood_quantiles.items()
                }
            }
        }
        save_path = os.path.join(self.evaluators[0].output_dir, "Benchmarks", f"{self.name}_benchmark_data.pt")
        torch.save(data_to_store, save_path)
        print(f"Benchmark data for plotting saved to: {save_path}")

        # ------------ Plotting ------------
        if self.cfg.problem.name == "Car" or self.cfg.problem.name == "Ahmed" :
            self._plot_3d(indices_to_plot, seed_idx)
        else:
            self._plot_2d(indices_to_plot, seed_idx)

    def _plot_2d(self, indices_to_plot=None, seed_idx=0):
        fig = plt.figure(figsize=(24, 18))
        for row_idx, (case_name, idx) in enumerate(indices_to_plot.items()):

            # Extract data (Move to CPU numpy)
            pts = self.X_test[seed_idx][idx].cpu().numpy()
            gt = self.Y_test[seed_idx][idx].cpu().numpy().flatten()
            pred = self.pred_mean[seed_idx][idx].cpu().numpy().flatten()
            std = self.pred_std[seed_idx][idx].cpu().numpy().flatten()
            abs_err = np.abs(gt - pred)

            # Common limits for this row (GT vs Pred)
            vmin = min(gt.min(), pred.min())
            vmax = max(gt.max(), pred.max())

            # Helper for 2D plotting
            def plot_cloud(col, values, title, cmap='viridis', v_range=None):
                ax = fig.add_subplot(3, 4, (row_idx * 4) + col + 1)

                if v_range:
                    sc = ax.scatter(pts[:,0], pts[:,1], c=values, s=20, cmap=cmap, vmin=v_range[0], vmax=v_range[1])
                else:
                    sc = ax.scatter(pts[:,0], pts[:,1], c=values, s=20, cmap=cmap)

                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_aspect('equal')
                ax.axis('off')
                return sc, ax

            # 1. Ground Truth
            p, ax = plot_cloud(0, gt, f"{case_name}: Ground Truth", v_range=(vmin, vmax))

            # 2. Prediction (Mean)
            p, ax = plot_cloud(1, pred, f"{case_name}: Pred Mean", v_range=(vmin, vmax))
            fig.colorbar(p, ax=ax, shrink=0.5)

            # 3. Uncertainty (Std)
            s_min, s_max = std.min(), std.max()
            p, ax = plot_cloud(2, std, f"{case_name}: Pred Std", cmap='plasma', v_range=(s_min, s_max))
            fig.colorbar(p, ax=ax, shrink=0.5)

            # 4. Error
            p, ax = plot_cloud(3, abs_err, f"{case_name}: Abs Error", cmap='inferno')
            fig.colorbar(p, ax=ax, shrink=0.5)

        plt.tight_layout()
        save_path = os.path.join(self.evaluators[0].output_dir, "Plots", f"{self.name}_2d_evaluation.png")
        plt.savefig(save_path, dpi=150)
        print(f"\nPlot saved successfully to: {save_path}")
        plt.close()

    def _plot_3d(self, indices_to_plot=None, seed_idx=0):
        fig = plt.figure(figsize=(24, 18))

        for row_idx, (case_name, idx) in enumerate(indices_to_plot.items()):

            # Extract data (Move to CPU numpy)
            pts = self.X_test[seed_idx][idx].cpu().numpy()
            gt = self.Y_test[seed_idx][idx].cpu().numpy().flatten()
            pred = self.pred_mean[seed_idx][idx].cpu().numpy().flatten()
            std = self.pred_std[seed_idx][idx].cpu().numpy().flatten()
            abs_err = np.abs(gt - pred)

            # Shared limits for Ground Truth and Prediction
            vmin = min(gt.min(), pred.min())
            vmax = max(gt.max(), pred.max())

            # Data ranges, to keep the true aspect ratio
            x_range = pts[:, 0].max() - pts[:, 0].min()
            y_range = pts[:, 1].max() - pts[:, 1].min()
            z_range = pts[:, 2].max() - pts[:, 2].min()

            # Helper for 3D plotting
            def plot_cloud_3d(col, values, title, cmap='viridis', v_range=None):
                # 3x4 grid: row_idx * 4 columns
                ax = fig.add_subplot(len(indices_to_plot), 4, (row_idx * 4) + col + 1, projection='3d')

                sc = ax.scatter(pts[:, 0], -pts[:, 1], pts[:, 2], c=values,
                                s=20, cmap=cmap, vmin=v_range[0] if v_range else None,
                                vmax=v_range[1] if v_range else None, alpha=0.8, edgecolors='none')

                ax.set_box_aspect((x_range, y_range, z_range))

                ax.set_title(f"{case_name}\n{title}", fontsize=12, fontweight='bold')

                # Clean up axes for a "point cloud" look
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

                # Set a consistent viewing angle
                ax.view_init(elev=30, azim=45)
                return sc, ax

            # 1. Ground Truth
            p, ax = plot_cloud_3d(0, gt, "Ground Truth", v_range=(vmin, vmax))

            # 2. Prediction (Mean)
            p, ax = plot_cloud_3d(1, pred, "Mean", v_range=(vmin, vmax))
            fig.colorbar(p, ax=ax, shrink=0.4, pad=0.1)

            # 3. Uncertainty (Std)
            p, ax = plot_cloud_3d(2, std, "Std", cmap='plasma')
            fig.colorbar(p, ax=ax, shrink=0.4, pad=0.1)

            # 4. Error
            p, ax = plot_cloud_3d(3, abs_err, "Abs Error", cmap='inferno')
            fig.colorbar(p, ax=ax, shrink=0.4, pad=0.1)

        plt.tight_layout()
        save_path = os.path.join(self.evaluators[0].output_dir, "Plots", f"{self.name}_3d_evaluation.png")
        plt.savefig(save_path, dpi=150)
        print(f"3D Plot saved successfully to: {save_path}")
        plt.close()
