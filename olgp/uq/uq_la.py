import os
import warnings
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from laplace import Laplace
from olgp.utils import set_seed
from olgp.baselines.transolver.evaluator import TransolverEvaluator
from olgp.uq.uq_evaluator import UQEvaluator
from olgp.uq.uq_utils import compute_energy_score_torch
from datetime import datetime

class UQLA(UQEvaluator):
    def __init__(self, cfg, pretrained_no_class, seeds=(0, 1, 2, 3, 4)):
        """Last-layer Laplace approximation."""
        super().__init__(cfg, pretrained_no_class, name="LA", seeds=seeds)
        self.laplace_approxs = []  # one Laplace approximation per seed

    def _extract_features(self, model, loader, device):
        """Runs backbone to get flattened features [Total_Points, Hidden] for Laplace."""
        model.eval()
        features_list, targets_list = [], []

        # Hook to capture input to the last layer
        captured = {}
        def hook_fn(m, i):
            captured['input'] = i[0]

        last_layer = self._get_last_layer(model)
        handle = last_layer.register_forward_pre_hook(hook_fn)

        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting Features", leave=False):
                if len(batch) == 2:  # training/validation loaders
                    x, y = batch
                else:  # test loader
                    x, _, y = batch

                x, y = x.to(device), y.to(device)
                model(x, None)

                # Flatten: [B, N, H] -> [B*N, H]
                feats = captured['input']
                feats_flat = feats.reshape(-1, feats.shape[-1])
                y_flat = y.reshape(-1, 1)

                features_list.append(feats_flat)
                targets_list.append(y_flat)

        handle.remove()
        return torch.cat(features_list), torch.cat(targets_list)

    def _get_last_layer(self, model):
        """Finds the last linear layer dynamically."""
        target_name = 'fc' if hasattr(model, 'fc') else 'output_layer'
        if hasattr(model, target_name):
            return getattr(model, target_name)

        # Otherwise use the last nn.Linear in the model
        last_linear = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                last_linear = module

        if last_linear is not None:
            return last_linear

        raise ValueError("Could not find last linear layer.")

    def _get_val_loader_raw(self, evaluator):
        train_path = os.path.join(evaluator.pre_path, rf'{evaluator.cfg.problem.name}_data.pt')
        dataset = torch.load(train_path, weights_only=True)

        X = dataset['pointcloud'].to(**evaluator.tkwargs)
        Y = dataset['output_features'].to(**evaluator.tkwargs)
        # Append local and global input features to the coordinates
        if dataset.get('local_features') is not None:
            X = torch.cat([X, dataset['local_features'].to(**evaluator.tkwargs)], dim=-1)
        if dataset.get('global_features') is not None:
            gf = dataset['global_features'].to(**evaluator.tkwargs)
            gf_expanded = gf.unsqueeze(1).expand(-1, X.shape[1], -1)
            X = torch.cat([X, gf_expanded], dim=-1)
        print(f"Original X shape: {X.shape}, Y shape: {Y.shape}")

        # Positional validation split
        N_val = self.cfg.problem.data.N_val
        N_test = self.cfg.problem.data.N_test
        X_val = X[-N_val-N_test:-N_test]
        Y_val = Y[-N_val-N_test:-N_test]

        # Normalize with the backbone's scalers
        X_norm = evaluator.scaler_X.transform(X_val)
        if Y_val.ndim == 2:
            Y_val = Y_val.unsqueeze(-1)
        Y_norm = evaluator.scaler_U.transform(Y_val)

        return DataLoader(
            TensorDataset(X_norm, Y_norm.squeeze(-1)),
            batch_size=32,
            shuffle=False
        )

    def _get_train_loader_for_hessian(self, evaluator):
        """Loads a subset of training data for Laplace curvature calculation."""
        train_path = os.path.join(evaluator.pre_path, rf'{evaluator.cfg.problem.name}_data.pt')
        dataset = torch.load(train_path, weights_only=True)
        X = dataset['pointcloud'].to(**evaluator.tkwargs)
        Y = dataset['output_features'].to(**evaluator.tkwargs)
        # Append local and global input features to the coordinates
        if dataset.get('local_features') is not None:
            X = torch.cat([X, dataset['local_features'].to(**evaluator.tkwargs)], dim=-1)
        if dataset.get('global_features') is not None:
            gf = dataset['global_features'].to(**evaluator.tkwargs)
            gf_expanded = gf.unsqueeze(1).expand(-1, X.shape[1], -1)
            X = torch.cat([X, gf_expanded], dim=-1)
        N_train = self.cfg.problem.data.N_train
        N_sub = min(N_train, 20000)
        X_sub = X[:N_sub]
        Y_sub = Y[:N_sub]
        if Y_sub.ndim == 2:
            Y_sub = Y_sub.unsqueeze(-1)
        # Normalize with the backbone's scalers
        X_norm = evaluator.scaler_X.transform(X_sub)
        Y_norm = evaluator.scaler_U.transform(Y_sub)
        # Targets are squeezed to [B, N] to match the model output
        return DataLoader(
            TensorDataset(X_norm, Y_norm.squeeze(-1)),
            batch_size=32,
            shuffle=True
        )

    def _fit(self):
        if self.fit:
            times = []
            memories = []
            for i, evaluator in enumerate(self.evaluators):
                start_time = datetime.now()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                print(f"\n{'='*60}")
                print(f"Processing LA Seed {self.seeds[i]} ({i+1}/{len(self.seeds)})")
                # 1. Setup
                set_seed(self.seeds[i])
                evaluator.setup(set_fixed_seed=True)
                # 2. Last-layer features of the training points
                train_loader_raw = self._get_train_loader_for_hessian(evaluator)
                print("  Extracting features from backbone...")
                X_feats, Y_targets = self._extract_features(evaluator.model, train_loader_raw, evaluator.device)

                # Random subset of points for the full-covariance Hessian
                max_laplace_points = 50000
                total_points = X_feats.shape[0]
                if total_points > max_laplace_points:
                    print(f"  Subsampling {total_points} points down to {max_laplace_points} for Laplace...")
                    rand_indices = torch.randperm(total_points)[:max_laplace_points]
                    X_feats = X_feats[rand_indices]
                    Y_targets = Y_targets[rand_indices]

                # 3. Copy of the last layer
                hessian_loader = DataLoader(TensorDataset(X_feats, Y_targets), batch_size=256, shuffle=True)
                orig_layer = self._get_last_layer(evaluator.model)
                la_model_linear = torch.nn.Linear(orig_layer.in_features, orig_layer.out_features)
                la_model_linear.load_state_dict(orig_layer.state_dict())
                la_model_linear = la_model_linear.to(evaluator.device)
                # 4. Fit, then tune the prior precision on validation points
                print(f"Fitting Laplace on {len(X_feats)} points...")
                la = Laplace(
                    la_model_linear,
                    likelihood='regression',
                    subset_of_weights='all',
                    hessian_structure='full',
                )
                la.fit(hessian_loader)
                print("Optimizing prior precision with grid search...")
                val_loader_raw = self._get_val_loader_raw(evaluator)
                X_val_feats, Y_val_targets = self._extract_features(evaluator.model, val_loader_raw, evaluator.device)
                max_val_points = 10000
                total_val = X_val_feats.shape[0]
                if total_val > max_val_points:
                    val_indices = torch.randperm(total_val)[:max_val_points]
                    X_val_feats = X_val_feats[val_indices]
                    Y_val_targets = Y_val_targets[val_indices]
                val_loader_features = DataLoader(
                    TensorDataset(X_val_feats, Y_val_targets),
                    batch_size=256,
                    shuffle=False
                )
                la.optimize_prior_precision(
                    method='gridsearch',
                    val_loader=val_loader_features,  # extracted features, not raw inputs
                    pred_type='glm',
                )
                # Save the fitted approximation
                self.laplace_approxs.append(la)
                la_path = os.path.join(self.evaluators[i].output_dir, "Models")
                torch.save(la.state_dict(), os.path.join(la_path, f"la_state_dict_seed_{self.seeds[i]}.pth"))
                end_time = datetime.now()
                times.append((end_time - start_time).total_seconds())
                fit_max_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
                memories.append(fit_max_mem_mb)

            return times, memories, True
        else:
            # Load pre-fitted Laplace approximations
            for i in tqdm(range(len(self.seeds))):
                # 1. Setup
                set_seed(self.seeds[i])
                self.evaluators[i].setup(set_fixed_seed=True)
                # 2. Load the fitted approximation
                orig_layer = self._get_last_layer(self.evaluators[i].model)
                la_model_linear = torch.nn.Linear(orig_layer.in_features, orig_layer.out_features)
                la_model_linear.load_state_dict(orig_layer.state_dict())
                la_model_linear = la_model_linear.to(self.evaluators[i].device)
                la = Laplace(
                    la_model_linear,
                    likelihood='regression',
                    subset_of_weights='all',
                    hessian_structure='full',
                )
                la_path = os.path.join(self.evaluators[i].output_dir, "Models")
                la.load_state_dict(torch.load(os.path.join(la_path, f"la_state_dict_seed_{self.seeds[i]}.pth"), map_location=self.evaluators[i].device, weights_only=True))
                self.laplace_approxs.append(la)
            print("Loaded pre-fitted Laplace approximations.")
            return None, None, False

    def _evaluate(self):
        pred_mean_list, pred_std_list, Y_test_list, X_test_list = [], [], [], []
        times, memories = [], []
        for i, evaluator in enumerate(self.evaluators):
            start_time = datetime.now()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            evaluator.model.eval()
            means, vars, targets, coords = [], [], [], []

            captured = {}
            def hook_fn(m, i):
                captured['input'] = i[0]

            handle = self._get_last_layer(evaluator.model).register_forward_pre_hook(hook_fn)
            with torch.no_grad():
                for batch in evaluator.test_loader:
                    if len(batch) == 3:
                        pos, _, y_gt = batch
                    else:
                        pos, y_gt = batch

                    x = pos.to(evaluator.device)

                    evaluator.model(x, None)
                    feats = captured['input'] # [B, N, H]
                    B, N, H = feats.shape

                    feats_flat = feats.reshape(-1, H)
                    f_mu_flat, f_var_flat = self.laplace_approxs[i](feats_flat)

                    f_mu = f_mu_flat.view(B, N)
                    f_var = f_var_flat.view(B, N)

                    # Back to physical units
                    pred_unnorm = evaluator.scaler_U.inverse(f_mu.unsqueeze(-1)).squeeze(-1)

                    target_unnorm = evaluator.scaler_U.inverse(y_gt.to(evaluator.device)).squeeze(-1)

                    scale = evaluator.scaler_U.std if hasattr(evaluator.scaler_U, 'std') else 1.0
                    if torch.is_tensor(scale):
                        scale = scale.to(f_var.device)
                    var_unnorm = f_var * (scale ** 2)

                    # Coordinates for plotting
                    if hasattr(evaluator.scaler_X, 'inverse'):
                        c = evaluator.scaler_X.inverse(x)[..., :self.cfg.problem.model.d]
                    else:
                        c = x[..., :self.cfg.problem.model.d]

                    means.append(pred_unnorm.cpu())
                    vars.append(var_unnorm.cpu())
                    targets.append(target_unnorm.cpu())
                    coords.append(c.cpu())

            handle.remove()

            # Concatenate the batches into full test-set tensors
            seed_mean = torch.cat(means, dim=0)       # [N_test, Points]
            seed_var = torch.cat(vars, dim=0)         # [N_test, Points]
            seed_std = torch.sqrt(seed_var).view_as(seed_mean)  # [N_test, Points]
            seed_targets = torch.cat(targets, dim=0)  # [N_test, Points]
            seed_coords = torch.cat(coords, dim=0)    # [N_test, Points, d]

            # Energy score from 50 Gaussian samples
            M = 50
            noise = torch.randn(M, *seed_mean.shape, device=seed_mean.device)
            samples = seed_mean.unsqueeze(0) + seed_std.unsqueeze(0) * noise

            seed_es = compute_energy_score_torch(samples, seed_targets)
            print(f"Energy Score for Seed {self.seeds[i]}: {seed_es:.4f}")

            if not hasattr(self, 'seed_energy_scores'):
                self.seed_energy_scores = []
            self.seed_energy_scores.append(seed_es)

            pred_mean_list.append(seed_mean.cpu())
            pred_std_list.append(seed_std.cpu())
            Y_test_list.append(seed_targets.cpu())
            X_test_list.append(seed_coords.cpu())

            end_time = datetime.now()
            times.append((end_time - start_time).total_seconds())
            fit_max_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            memories.append(fit_max_mem_mb)

        # Stack results across seeds
        self.pred_mean = torch.stack(pred_mean_list) # [Seeds, Samples, N]
        self.pred_std = torch.stack(pred_std_list)
        self.Y_test = torch.stack(Y_test_list)
        self.X_test = torch.stack(X_test_list)

        return times, memories, self.seed_energy_scores

@hydra.main(version_base=None, config_path="../../configs", config_name="config_transolver")
def main(cfg: DictConfig):
    warnings.filterwarnings("ignore", category=UserWarning)
    evaluator = UQLA(cfg, pretrained_no_class=TransolverEvaluator, seeds=[0, 1, 2, 3, 4])
    evaluator.run()

if __name__ == "__main__":
    main()
