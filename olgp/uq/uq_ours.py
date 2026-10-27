import copy
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import gpytorch
from linear_operator.utils.errors import NotPSDError
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from datetime import datetime
from olgp.utils import set_seed
from olgp.baselines.transolver.evaluator import TransolverEvaluator
from olgp.uq.uq_evaluator import UQEvaluator
from olgp.uq.uq_ours_model import REEFGP, NNHeteroscedasticLikelihood
from olgp.uq.uq_utils import compute_energy_score_torch
import warnings

class UQREEFGP(UQEvaluator):
    def __init__(self, cfg, pretrained_no_class, seeds=(0, 1, 2, 3, 4), fit=False):
        """ REEF-GP Method """
        name = "REEFGP"
        super().__init__(cfg, pretrained_no_class, name=name, seeds=seeds, fit=fit)
        self.our_gp_models = []  # one fitted GP per seed
        self.subset_batch_size = 5000  # Stochastic subset size for train
        self.train_epochs = self.cfg.problem.uq.train_epochs
        self.support_size = self.subset_batch_size   # Stochastic subset size for inference
        self.eval_chunk_size = 10000  # points per GP prediction call
        self.num_experts = 5

    def _extract_features_and_residuals(self, model, loader, device):
        """Runs the frozen backbone and returns the GP inputs and targets.

        Inputs are [coords, embeddings of the selected layers, backbone prediction].
        Targets are the ground-truth field (the backbone prediction is the GP mean).
        """
        if not hasattr(self, 'layer_names'):
            self._auto_select_layers(model, loader, device)

        model.eval()
        feats_list, resid_list = [], []

        # Forward hooks that capture the outputs of the selected layers
        captured = {}
        hooks = []

        def get_hook(layer_name):
            def hook_fn(module, input, output):
                # Some blocks return tuples; keep the hidden state
                if isinstance(output, tuple):
                    captured[layer_name] = output[0]
                else:
                    captured[layer_name] = output
            return hook_fn

        for name in self.layer_names:
            submodule = model.get_submodule(name)
            hook = submodule.register_forward_hook(get_hook(name))
            hooks.append(hook)

        with torch.no_grad():
            for batch in loader:
                # Batches are (x, y) or (x, x, y): x already holds coords + input features
                x, y_gt = batch[0], batch[-1]
                x = x.to(device)
                y_gt = y_gt.to(device)

                y_pred_norm = model(x, None)

                # Concatenate the embeddings along the feature dimension
                emb_list = [captured[name] for name in self.layer_names]
                embeddings = torch.cat(emb_list, dim=-1) # [B, N, H_total]
                coords = x  # normalized coords and input features

                # The GP is fitted on the full field y with the frozen NN prediction as its mean
                # function (NeuralMean). This is equivalent to the paper's zero-mean GP on the
                # residual delta = y - NN(x). "resid_mu"/"resid_std" below are the stats of y.
                if y_gt.ndim == 2:
                    y_gt = y_gt.unsqueeze(-1)
                target_full = y_gt.view_as(y_pred_norm)

                # Backbone prediction, used as the GP mean
                nn_pred = y_pred_norm.view_as(target_full)

                coords_flat = coords.reshape(-1, coords.shape[-1])
                embed_flat = embeddings.reshape(-1, embeddings.shape[-1])
                nn_pred_flat = nn_pred.reshape(-1, 1) # [B*N, 1]
                target_flat = target_full.reshape(-1)

                # GP input layout: [coords, embeddings, nn_pred]
                combined_feats = torch.cat([coords_flat, embed_flat, nn_pred_flat], dim=-1)

                feats_list.append(combined_feats.cpu())
                resid_list.append(target_flat.cpu())

        for hook in hooks:
            hook.remove()
        return torch.cat(feats_list), torch.cat(resid_list)

    def _auto_select_layers(self, model, loader, device):
        """Selects the first layer, the last layer, and the middle layer that is least
        redundant with both (lowest summed linear CKA), using one batch of training data.
        """
        print("\n--- Auto-selecting most informative middle layer via CKA ---")
        model.eval()

        # Candidates: every block, and the MLP of the last block (before the output head)
        n_layers = self.cfg.problem.model.n_layers
        candidate_layers = [f'blocks.{i}' for i in range(n_layers - 1)] + [f'blocks.{n_layers - 1}.mlp']

        captured = {}
        hooks = []

        def get_hook(name):
            def hook_fn(module, input, output):
                captured[name] = output[0] if isinstance(output, tuple) else output
            return hook_fn

        for name in candidate_layers:
            submodule = model.get_submodule(name)
            hooks.append(submodule.register_forward_hook(get_hook(name)))

        # One batch is enough for CKA
        with torch.no_grad():
            for batch in loader:
                x = batch[0]
                model(x.to(device), None)
                break

        for h in hooks:
            h.remove()

        # Flatten and subsample points to keep CKA cheap
        flat_embs = {}
        for name, emb in captured.items():
            emb_flat = emb.reshape(-1, emb.shape[-1])
            torch.manual_seed(42)
            idx = torch.randperm(emb_flat.shape[0])[:5000]
            flat_embs[name] = emb_flat[idx]

        # Linear CKA of each middle layer against the first and last layers
        start_name, end_name = candidate_layers[0], candidate_layers[-1]

        def get_centered_and_norm(tensor):
            t_c = tensor - tensor.mean(dim=0)
            return t_c, torch.norm(t_c.T @ t_c)

        X_start_c, norm_start = get_centered_and_norm(flat_embs[start_name])
        X_end_c, norm_end = get_centered_and_norm(flat_embs[end_name])

        best_mid_layer = None
        min_combined_sim = float('inf')

        print(f"{'Layer':>12} | Sim to {start_name} | Sim to {end_name} | Sum (Redundancy)")
        print("-" * 65)

        for i in range(1, len(candidate_layers) - 1):
            name = candidate_layers[i]
            Y_c, norm_Y = get_centered_and_norm(flat_embs[name])

            dot_start = torch.norm(Y_c.T @ X_start_c)**2
            sim_start = (dot_start / (norm_start * norm_Y)).item()

            dot_end = torch.norm(Y_c.T @ X_end_c)**2
            sim_end = (dot_end / (norm_end * norm_Y)).item()

            combined = sim_start + sim_end
            print(f"{name:>12} | {sim_start:15.3f} | {sim_end:15.3f} | {combined:16.3f}")

            if combined < min_combined_sim:
                min_combined_sim = combined
                best_mid_layer = name

        print("-" * 65)

        self.layer_names = [start_name, best_mid_layer, end_name]
        print(f"Auto-selected layers for GP: {self.layer_names}\n")

    def _get_train_loader(self, evaluator):
        """Loads a subset of training data for GP fitting."""
        train_path = os.path.join(evaluator.pre_path, rf'{evaluator.cfg.problem.name}_data.pt')
        dataset = torch.load(train_path, weights_only=True)

        X = dataset['pointcloud'].to(**evaluator.tkwargs)  # [B, N, d]
        Y = dataset['output_features'].to(**evaluator.tkwargs)
        # Append local and global input features to the coordinates
        if dataset.get('local_features') is not None:
            X = torch.cat([X, dataset['local_features'].to(**evaluator.tkwargs)], dim=-1)
        if dataset.get('global_features') is not None:
            gf = dataset['global_features'].to(**evaluator.tkwargs)
            gf_expanded = gf.unsqueeze(1).expand(-1, X.shape[1], -1)
            X = torch.cat([X, gf_expanded], dim=-1)
        print(f"Original X shape: {X.shape}, Y shape: {Y.shape}")

        # Positional splits: train first, then validation, then test
        N_train = self.cfg.problem.data.N_train
        N_val = self.cfg.problem.data.N_val
        N_test = self.cfg.problem.data.N_test

        X_train = X[:N_train] # [N_sub, N_points, Feats]
        Y_train = Y[:N_train] # [N_sub, N_points]
        X_val = X[-N_val-N_test:-N_test]
        Y_val = Y[-N_val-N_test:-N_test]

        # Normalize with the backbone's scalers
        X_train_norm = evaluator.scaler_X.transform(X_train)
        if Y_train.ndim == 2:
            Y_train = Y_train.unsqueeze(-1) # [N_sub, N_points, 1]
        Y_train_norm = evaluator.scaler_U.transform(Y_train)

        # Targets are squeezed to [B, N] to match the Transolver loaders
        Y_final = Y_train_norm.squeeze(-1).squeeze(0)
        if X_train_norm.size(0) != Y_final.size(0):
            raise ValueError(f"Size mismatch: X {X_train_norm.shape} vs Y {Y_final.shape}")
        X_val_norm = evaluator.scaler_X.transform(X_val)
        if Y_val.ndim == 2:
            Y_val = Y_val.unsqueeze(-1)
        Y_val_norm = evaluator.scaler_U.transform(Y_val)
        Y_val_final = Y_val_norm.squeeze(-1).squeeze(0)
        if X_val_norm.size(0) != Y_val_final.size(0):
            raise ValueError(f"Size mismatch: X_val {X_val_norm.shape} vs Y_val {Y_val_final.shape}")

        return DataLoader(
            TensorDataset(X_train_norm, Y_final),
            batch_size=32,
            shuffle=False
        ), DataLoader(
            TensorDataset(X_val_norm, Y_val_final),
            batch_size=32,
            shuffle=False
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
                print(f"Processing GP Seed {self.seeds[i]} ({i+1}/{len(self.seeds)})")
                # 1. Load the backbone
                set_seed(self.seeds[i])
                evaluator.setup(set_fixed_seed=True)
                # 2. Extract GP inputs and targets
                loader, val_loader = self._get_train_loader(evaluator)
                X_all, Y_resid_all = self._extract_features_and_residuals(
                    evaluator.model, loader, evaluator.device
                )
                X_val_all, Y_val_resid_all = self._extract_features_and_residuals(
                    evaluator.model, val_loader, evaluator.device
                )
                X_train = X_all
                Y_train = Y_resid_all
                X_val = X_val_all
                Y_val = Y_val_resid_all

                # Standardize the targets and the backbone prediction column
                self.resid_mu = Y_train.mean()
                self.resid_std = Y_train.std()
                if self.resid_std < 1e-9:
                    self.resid_std = torch.tensor(1.0, device=evaluator.device)
                Y_train_gp = (Y_train - self.resid_mu) / self.resid_std
                Y_val_gp = (Y_val - self.resid_mu) / self.resid_std
                X_train[..., -1] = (X_train[..., -1] - self.resid_mu) / self.resid_std
                X_val[..., -1]   = (X_val[..., -1] - self.resid_mu) / self.resid_std
                # 3. Fit the GP
                emb_dim = X_train.shape[1] - self.cfg.problem.model.d - self.cfg.problem.model.fun_dim - 1
                if self.cfg.problem.uq.noise_model == "heteroskedastic":
                    print("Using Heteroskedastic Likelihood")
                    likelihood = NNHeteroscedasticLikelihood(self.cfg, embedding_dim=emb_dim, d=self.cfg.problem.model.d, fun_dim=self.cfg.problem.model.fun_dim).to(evaluator.device)
                else:
                    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(evaluator.device)
                    likelihood.noise_covar.noise = 1e-4
                    likelihood.noise_covar.raw_noise.requires_grad_(True)
                gp_model = REEFGP(self.cfg, self.cfg.problem.name, None, None, likelihood, embedding_dim=emb_dim, d=self.cfg.problem.model.d, fun_dim=self.cfg.problem.model.fun_dim).to(evaluator.device)

                gp_model.train()
                likelihood.train()
                # The warp and noise networks get their own learning rate and weight decay
                emb_params = list(gp_model.k_embedding.parameters())
                if self.cfg.problem.uq.noise_model == "heteroskedastic":
                    noise_params = list(gp_model.likelihood.parameters())
                    ann_params = emb_params + noise_params
                else:
                    ann_params = emb_params
                ann_ids = set(map(id, ann_params))
                rest_params = [p for p in gp_model.parameters() if id(p) not in ann_ids]

                optimizer = torch.optim.AdamW([
                    {'params': ann_params, 'lr': self.cfg.problem.uq.nn_lr, 'weight_decay': self.cfg.problem.uq.nn_weight_decay},
                    {'params': rest_params, 'lr': self.cfg.problem.uq.gp_lr}
                ])
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

                pbar = tqdm(range(self.train_epochs), desc="Training")
                val_loss = float('inf')
                val_step = 20
                best_state_dict = None
                best_val_metric = val_loss
                no_improvement_evals = 0  # counted in validation steps
                history = {"train_loss": [], "val_loss": [], "epochs": []}

                self._used_cg_fallback = False
                for epoch in pbar:
                    if no_improvement_evals >= self.cfg.problem.uq.convergence_patience:
                        print(f"Early stopping at epoch {epoch}: no val improvement for {no_improvement_evals * val_step} epochs.")
                        break

                    optimizer.zero_grad()
                    indices = torch.randperm(X_train.shape[0])
                    idx_full = indices[:self.subset_batch_size]
                    X_full, Y_full = X_train[idx_full].to(evaluator.device), Y_train_gp[idx_full].to(evaluator.device)
                    gp_model.set_train_data(inputs=X_full, targets=Y_full, strict=False)
                    try:
                        with gpytorch.settings.max_cholesky_size(1000000), \
                            gpytorch.settings.cholesky_jitter(1e-6):
                            loss = -mll(gp_model(X_full), Y_full)
                            loss.backward()
                    except NotPSDError:
                        # Fall back to CG by disabling Cholesky
                        if not self._used_cg_fallback:
                            print(f"  [epoch {epoch}] Cholesky failed; falling back to CG for the rest of training.")
                            self._used_cg_fallback = True
                        optimizer.zero_grad()  # discard gradients from the failed attempt
                        with gpytorch.settings.max_cholesky_size(0), \
                            gpytorch.settings.max_cg_iterations(1000):
                            loss = -mll(gp_model(X_full), Y_full)
                            loss.backward()
                    torch.nn.utils.clip_grad_norm_(gp_model.parameters(), max_norm=10.0)
                    optimizer.step()

                    # Validate every val_step epochs
                    if epoch % val_step == 0:
                        gp_model.eval()
                        likelihood.eval()
                        val_losses = []
                        val_batch_size = self.eval_chunk_size
                        with torch.no_grad(), gpytorch.settings.fast_pred_var():
                            for c in range(0, X_val.shape[0], val_batch_size):
                                X_batch = X_val[c : c + val_batch_size].to(evaluator.device)
                                Y_batch = Y_val_gp[c : c + val_batch_size].to(evaluator.device)
                                observed_pred = likelihood(gp_model(X_batch))
                                batch_val_loss = -torch.distributions.Normal(
                                    observed_pred.mean, observed_pred.stddev
                                ).log_prob(Y_batch).mean()
                                val_losses.append(batch_val_loss.item())
                        val_loss = sum(val_losses) / len(val_losses)
                        history["train_loss"].append(loss.item())
                        history["val_loss"].append(val_loss)
                        history["epochs"].append(epoch)

                        # Early stopping on the validation NLL
                        if val_loss < best_val_metric:
                            best_val_metric = val_loss
                            best_state_dict = copy.deepcopy(gp_model.state_dict())
                            no_improvement_evals = 0
                        else:
                            no_improvement_evals += 1

                        gp_model.train()
                        likelihood.train()

                    pbar.set_postfix({'loss': loss.item(), 'val_loss': val_loss})

                # Keep the best checkpoint
                if best_state_dict is not None:
                    gp_model.load_state_dict(best_state_dict)
                self.our_gp_models.append(gp_model)
                models_dir = os.path.join(self.evaluators[i].output_dir, "Models")
                torch.save(best_state_dict, os.path.join(models_dir, f"gp_state_dict_seed_{self.seeds[i]}.pth"))
                end_time = datetime.now()
                times.append((end_time - start_time).total_seconds())
                fit_max_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
                memories.append(fit_max_mem_mb)

                torch.save(history, os.path.join(models_dir, f"history_seed_{self.seeds[i]}.pth"))
                if not hasattr(self, 'training_histories'):
                    self.training_histories = {}
                self.training_histories[self.seeds[i]] = history

            return times, memories, True
        else:
            # Load pre-fitted GP approximations
            for i in tqdm(range(len(self.seeds))):
                evaluator = self.evaluators[i]
                # 1. Setup
                set_seed(self.seeds[i])
                evaluator.setup(set_fixed_seed=True)
                # 2. Infer the embedding size from the features
                loader, val_loader = self._get_train_loader(evaluator)
                X_all, Y_resid_all = self._extract_features_and_residuals(
                    evaluator.model, loader, evaluator.device
                )
                X_train = X_all
                emb_dim = X_train.shape[1] - self.cfg.problem.model.d - self.cfg.problem.model.fun_dim - 1
                # 3. Load GP
                if self.cfg.problem.uq.noise_model == "heteroskedastic":
                    print("Using Heteroskedastic Likelihood")
                    likelihood = NNHeteroscedasticLikelihood(self.cfg, embedding_dim=emb_dim, d=self.cfg.problem.model.d, fun_dim=self.cfg.problem.model.fun_dim).to(evaluator.device)
                else:
                    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(evaluator.device)
                    likelihood.noise_covar.noise = 1e-4
                    likelihood.noise_covar.raw_noise.requires_grad_(True)
                gp_model = REEFGP(self.cfg, self.cfg.problem.name, None, None, likelihood, embedding_dim=emb_dim, d=self.cfg.problem.model.d, fun_dim=self.cfg.problem.model.fun_dim).to(evaluator.device)
                gp_path = os.path.join(evaluator.output_dir, "Models")
                gp_model.load_state_dict(torch.load(os.path.join(gp_path, f"gp_state_dict_seed_{self.seeds[i]}.pth"), map_location=evaluator.device, weights_only=True))
                self.our_gp_models.append(gp_model)
            print("Loaded pre-fitted GP approximations.")
            return None, None, False

    @torch.inference_mode()
    def _evaluate(self):
        agg_method = self.cfg.problem.uq.aggregation.lower()

        print("\nStarting Ensemble Inference on Test Set...")
        print("=="*50)
        pred_mean_list, pred_std_list, Y_test_list, X_test_list = [], [], [], []
        times, memories = [], []
        self.seed_energy_scores = []

        for i, evaluator in enumerate(self.evaluators):
            # Start metrics
            start_time = datetime.now()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

            evaluator.model.eval()
            gp_model = self.our_gp_models[i]
            gp_model.eval()
            gp_model.likelihood.eval()

            # 1. Training features: standardization statistics and expert support sets
            loader, val_loader = self._get_train_loader(evaluator)
            X_all, Y_resid_all = self._extract_features_and_residuals(
                evaluator.model, loader, evaluator.device
            )
            X_train = X_all.clone()
            Y_train = Y_resid_all
            self.resid_mu = Y_train.mean()
            self.resid_std = Y_train.std()
            Y_train_gp = (Y_train - self.resid_mu) / self.resid_std
            X_train[..., -1] = (X_train[..., -1] - self.resid_mu) / self.resid_std

            support_sets = []
            for k in range(self.num_experts):
                torch.manual_seed(self.seeds[i] + (100 + k))
                idx_k = torch.randperm(X_train.shape[0])[:self.support_size]
                X_supp = X_train[idx_k].to(evaluator.device)
                Y_supp = Y_train_gp[idx_k].to(evaluator.device)
                support_sets.append((X_supp, Y_supp))

            # 2. Hooks on the selected layers
            captured = {}
            hooks = []
            def get_hook(layer_name):
                def hook_fn(module, input, output):
                    captured[layer_name] = output[0] if isinstance(output, tuple) else output
                return hook_fn
            for name in self.layer_names:
                submodule = evaluator.model.get_submodule(name)
                hooks.append(submodule.register_forward_hook(get_hook(name)))

            # 3a. Extract the GP inputs of all test batches once and cache them on the CPU
            print("  Pass 1: extracting embeddings and NN predictions...")
            batch_inputs = []  # list of GP-input tensors, one per batch
            batch_targets = []
            batch_coords = []
            batch_shapes = []  # (B, N) per batch

            for batch in tqdm(evaluator.test_loader, desc="Embedding extraction", leave=False):
                x, y_gt = batch[0], batch[-1]
                x = x.to(evaluator.device)
                y_gt = y_gt.to(evaluator.device)

                y_nn_norm = evaluator.model(x, None)
                if y_nn_norm.ndim == 3:
                    if y_nn_norm.shape[-1] == y_nn_norm.shape[1]:
                        y_nn_norm = y_nn_norm.mean(dim=-1)
                    elif y_nn_norm.shape[-1] == 1:
                        y_nn_norm = y_nn_norm.squeeze(-1)

                emb_list = [captured[name] for name in self.layer_names]
                embeddings = torch.cat(emb_list, dim=-1)
                B, N, H = embeddings.shape

                coords_flat = x.reshape(-1, x.shape[-1])
                embed_flat = embeddings.reshape(-1, H)
                nn_pred_flat = y_nn_norm.reshape(-1, 1)
                nn_pred_flat = (nn_pred_flat - self.resid_mu) / self.resid_std
                gp_input = torch.cat([coords_flat, embed_flat, nn_pred_flat], dim=-1)

                batch_inputs.append(gp_input.cpu())  # cache to CPU
                target_real = evaluator.scaler_U.inverse(y_gt.unsqueeze(-1) if y_gt.ndim == 2 else y_gt).squeeze(-1)
                c_unnorm = evaluator.scaler_X.inverse(x)
                batch_targets.append(target_real.cpu())
                batch_coords.append(c_unnorm.cpu())
                batch_shapes.append((B, N))

                del gp_input, embeddings, embed_flat, coords_flat, nn_pred_flat, y_nn_norm
                captured.clear()

            for h in hooks:
                h.remove()

            # 3b. Predict with each of the K experts on the cached inputs
            print("  Pass 2: K-expert GP predictions on cached embeddings...")
            all_mu_chunks = [[] for _ in range(self.num_experts)]
            all_var_chunks = [[] for _ in range(self.num_experts)]
            all_noise_chunks = []

            # The noise variance depends only on the input, not on the support set
            gp_model.set_train_data(inputs=support_sets[0][0], targets=support_sets[0][1], strict=False)
            print("  Computing noise variance once across all batches...")
            for b_idx, gp_input_cpu in enumerate(tqdm(batch_inputs, desc="Noise", leave=False)):
                B, N = batch_shapes[b_idx]
                noise_chunk_flat = torch.zeros(B * N)
                for c in range(0, gp_input_cpu.shape[0], self.eval_chunk_size):
                    end = min(c + self.eval_chunk_size, gp_input_cpu.shape[0])
                    chunk = gp_input_cpu[c:end].to(evaluator.device)
                    noise_covar = gp_model.likelihood.noise_covar(chunk)
                    noise_chunk_flat[c:end] = noise_covar.diagonal().cpu()
                all_noise_chunks.append(noise_chunk_flat.view(B, N))

            # Outer loop over experts so set_train_data is called only K times
            for k, (X_supp, Y_supp) in enumerate(support_sets):
                gp_model.set_train_data(inputs=X_supp, targets=Y_supp, strict=False)
                print(f"  Expert {k+1}/{self.num_experts}...")
                for b_idx, gp_input_cpu in enumerate(tqdm(batch_inputs, desc=f"Expert {k+1}", leave=False)):
                    B, N = batch_shapes[b_idx]
                    mu_pieces = []
                    var_pieces = []
                    for c in range(0, gp_input_cpu.shape[0], self.eval_chunk_size):
                        end = min(c + self.eval_chunk_size, gp_input_cpu.shape[0])
                        chunk = gp_input_cpu[c:end].to(evaluator.device)
                        with gpytorch.settings.fast_pred_var():
                            out = gp_model(chunk)
                        mu_pieces.append(out.mean.cpu())
                        var_pieces.append(out.variance.cpu())
                    mu_k = torch.cat(mu_pieces).view(B, N)
                    var_k = torch.cat(var_pieces).view(B, N)
                    all_mu_chunks[k].append(mu_k)
                    all_var_chunks[k].append(var_k)

            del batch_inputs

            # 4. Concatenate batches
            all_mu = torch.stack([torch.cat(chunks, dim=0) for chunks in all_mu_chunks])
            all_latent_var = torch.stack([torch.cat(chunks, dim=0) for chunks in all_var_chunks])
            noise_var_std = torch.cat(all_noise_chunks, dim=0)
            final_targets = torch.cat(batch_targets, dim=0)
            final_coords = torch.cat(batch_coords, dim=0)

            # 5. Map back to physical units: undo the GP standardization, then the backbone's scaler
            scale_A = self.resid_std
            scale_B = evaluator.scaler_U.std if hasattr(evaluator.scaler_U, 'std') else 1.0
            if torch.is_tensor(scale_B):
                scale_B = scale_B.cpu()
            total_scale_sq = (scale_A ** 2) * (scale_B ** 2)

            # Means: from standardized -> Transolver space -> physical
            all_mu_transolver = all_mu * self.resid_std + self.resid_mu  # [K, total_B, N]
            # Inverse-transform per expert
            all_mu_phys = []
            for k in range(self.num_experts):
                mu_k = all_mu_transolver[k]  # [total_B, N]
                mu_k_phys = evaluator.scaler_U.inverse(mu_k.unsqueeze(-1)).squeeze(-1)
                all_mu_phys.append(mu_k_phys)
            all_mu_phys = torch.stack(all_mu_phys)  # [K, total_B, N]

            all_latent_var_phys = all_latent_var * total_scale_sq
            noise_var_phys = noise_var_std * total_scale_sq

            # 6. Aggregate the latent experts, then add the heteroscedastic noise
            mu, agg_latent_var = self._aggregate_predictions(all_mu_phys, all_latent_var_phys, agg_method)
            final_var = agg_latent_var + noise_var_phys

            final_mu = mu.squeeze()
            final_std = torch.sqrt(final_var.squeeze())
            final_targets = final_targets.squeeze()
            final_coords = final_coords.squeeze()

            # Sanity checks
            if final_mu.dim() > final_targets.dim() and final_mu.shape[0] == self.num_experts:
                raise RuntimeError("Aggregated mean still has expert dimension; bug in _aggregate_predictions.")
            if final_mu.numel() != final_targets.numel():
                raise RuntimeError(
                    f"Shape mismatch: pred {final_mu.shape} vs targets {final_targets.shape}"
                )

            # 7. Energy score
            seed_mean = final_mu
            seed_std = final_std
            seed_targets = final_targets
            M = 50
            noise = torch.randn(M, *seed_mean.shape)
            samples = seed_mean.unsqueeze(0) + seed_std.unsqueeze(0) * noise
            seed_es = compute_energy_score_torch(samples, seed_targets)
            print(f"Energy Score for Seed {self.seeds[i]}: {seed_es:.4f}")
            self.seed_energy_scores.append(seed_es)

            pred_mean_list.append(final_mu)
            pred_std_list.append(final_std)
            Y_test_list.append(final_targets)
            X_test_list.append(final_coords)

            end_time = datetime.now()
            times.append((end_time - start_time).total_seconds())
            fit_max_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            memories.append(fit_max_mem_mb)

        self.pred_mean = torch.stack(pred_mean_list)
        self.pred_std = torch.stack(pred_std_list)
        self.Y_test = torch.stack(Y_test_list)
        self.X_test = torch.stack(X_test_list)
        return times, memories, self.seed_energy_scores

    def _aggregate_predictions(self, all_mu, all_var, method):
        """
        Aggregates predictions from K experts.
        all_mu, all_var: shape [K, ...]
        """
        K = all_mu.shape[0]

        # Guard against zero variance numerical issues
        eps = 1e-8
        all_var = torch.clamp(all_var, min=eps)
        precisions = 1.0 / all_var

        if method == "moe":
            # Standard equally-weighted Mixture of Experts
            mu = all_mu.mean(dim=0)
            avg_internal_var = all_var.mean(dim=0)
            expert_disagreement = all_mu.var(dim=0, unbiased=False)
            var = avg_internal_var + expert_disagreement

        elif method == "gpoe":
            # Generalized Product of Experts
            # Beta weights sum to 1 to ensure fallback to prior
            beta = 1.0 / K
            agg_precision = (beta * precisions).sum(dim=0)
            var = 1.0 / agg_precision
            mu = var * (beta * precisions * all_mu).sum(dim=0)

        else:
            raise ValueError(f"Unknown aggregation method: {method}. Use 'gpoe' or 'moe'.")

        return mu, var

@hydra.main(version_base=None, config_path="../../configs", config_name="config_transolver")
def main(cfg: DictConfig):
    warnings.filterwarnings("ignore", category=UserWarning)
    evaluator = UQREEFGP(cfg, pretrained_no_class=TransolverEvaluator, seeds=[0, 1, 2, 3, 4], fit=cfg.problem.uq.fit)
    evaluator.run()

if __name__ == "__main__":
    main()
