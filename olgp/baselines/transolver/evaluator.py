import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from olgp.baselines.transolver.src.model_dict import get_model
from olgp.normalize import StandardScaler
from olgp.baselines.base_evaluator import BaselineEvaluator

class TransolverEvaluator(BaselineEvaluator):
    def __init__(self, cfg):
        super().__init__(cfg)
    def _load_model_impl(self):
        """Builds model and restores state + scalers."""
        # Load checkpoint
        model_path = os.path.join(self.output_dir, "Models/best_model.pth")
        self.logger.info(f"Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        # Build model architecture (must match training config exactly)
        model_name = 'Transolver_Irregular_Mesh'
        self.model = get_model(model_name)(
            space_dim=self.cfg.problem.model.d,
            n_layers=self.cfg.problem.model.n_layers,
            n_hidden=self.cfg.problem.model.n_hidden,
            dropout=self.cfg.problem.model.dropout,
            n_head=self.cfg.problem.model.n_heads,
            Time_Input=False,
            mlp_ratio=self.cfg.problem.model.mlp_ratio,
            fun_dim=self.cfg.problem.model.fun_dim,
            out_dim=self.cfg.problem.model.out_dim,
            slice_num=self.cfg.problem.model.slice_num,
            ref=self.cfg.problem.model.ref,
            unified_pos=self.cfg.problem.model.unified_pos,
            act=self.cfg.problem.model.act,
        ).to(**self.tkwargs)
        # Load weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Restore scalers (manually initializing with saved stats)
        self.scaler_X = StandardScaler(mean=checkpoint["X_mean"], std=checkpoint["X_std"])
        self.scaler_U = StandardScaler(mean=checkpoint["U_mean"], std=checkpoint["U_std"])

        self.logger.info("Transolver Model and scalers restored successfully.")

    def _load_data_impl(self):
        """Loads test data -> normalizes -> prepares Loader."""
        # Load dataset
        if self.cfg.problem.inference.split == "test":
            self.logger.info(f"Loading and preparing test dataset: {os.path.join(self.pre_path_test, rf'{self.cfg.problem.name}_data.pt')}")
            dataset = torch.load(os.path.join(self.pre_path_test, rf'{self.cfg.problem.name}_data.pt'), weights_only=True)
        elif self.cfg.problem.inference.split == "train":
            self.logger.info(f"Loading and preparing train dataset: {os.path.join(self.pre_path, rf'{self.cfg.problem.name}_data.pt')}")
            dataset = torch.load(os.path.join(self.pre_path, rf'{self.cfg.problem.name}_data.pt'), weights_only=True)
        else:
            raise ValueError(f"Invalid inference split: {self.cfg.problem.inference.split}. Must be 'train' or 'test'.")
        pointcloud = dataset['pointcloud'].to(**self.tkwargs) # [Nsamples, Npoints, 2]
        output_features = dataset['output_features'].to(**self.tkwargs) # [Nsamples, Npoints, Nfeatures]
        local_features = dataset['local_features'].to(**self.tkwargs) if dataset['local_features'] is not None else None # [Nsamples, Npoints, Nfeatures]
        global_features = dataset['global_features'].to(**self.tkwargs) if dataset['global_features'] is not None else None # [Nsamples, Nfeatures]
        X_data = pointcloud # [Nsamples, Npoints, 3]
        if local_features is not None:
            X_data = torch.cat([pointcloud, local_features], dim=-1) # [Nsamples, Npoints, Nfeatures]
        if global_features is not None:
            gf_expanded = global_features.unsqueeze(1).expand(-1, X_data.shape[1], -1)
            # Concatenate
            X_data = torch.cat([X_data, gf_expanded], dim=-1) # Result shape: [Nsamples, Npoints, Nfeatures + D]
        U_data = output_features # [Nsamples, Npoints, Nfeatures]
        self.logger.info(f"Input Shape: {X_data.shape}, Output Shape: {U_data.shape}")
        # Split
        if self.cfg.problem.inference.split == "test":
            N_test = self.cfg.problem.data.N_test
            X_test = X_data[-N_test:, ...] # [Nsamples, Npoints, Nfeatures]
            U_test = U_data[-N_test:, ...].unsqueeze(-1) # [Nsamples, Npoints, 1]
            self.logger.debug(f"Test Input Shape: {X_test.shape}, Test Output Shape: {U_test.shape}")
        elif self.cfg.problem.inference.split == "train":
            N_train = self.cfg.problem.data.N_train
            X_test = X_data[:N_train, ...] # [Nsamples, Npoints, Nfeatures]
            U_test = U_data[:N_train, ...].unsqueeze(-1) # [Nsamples, Npoints, 1]
            self.logger.debug(f"Train Input Shape: {X_test.shape}, Train Output Shape: {U_test.shape}")
        # Normalize
        X_test_norm = self.scaler_X.transform(X_test)
        U_test_norm = self.scaler_U.transform(U_test)
        # DataLoaders
        print(f"Shape of X_test_norm: {X_test_norm.shape}, Shape of U_test_norm: {U_test_norm.shape}")
        test_loader = DataLoader(
            TensorDataset(X_test_norm, X_test_norm, U_test_norm), # Transolver passes x_train twice
            batch_size=1,
            shuffle=False,
        )

        # Set final values
        self.test_loader = test_loader

    def _enable_dropout_only(self):
        """
        Forces only Dropout layers to be in training mode.
        Ensures BatchNorm layers remain frozen (eval mode).
        """
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def compute_inference(self, batch):
        """
        Unpacks custom batch, runs forward pass, returns unnormalized results + coords + extras.
        Batch structure from _load_data_impl: (x_norm, u_raw, pos_raw, feats_raw)
        """
        # Unpack batch
        pos, fx, y = batch
        x, _, y = pos.to(self.device), fx.to(self.device), y.to(self.device)
        # Forward Pass
        if self.cfg.ablation.name == "dropout" or self.cfg.ablation.name == "pno":
            self._enable_dropout_only()  # Enable dropout for MC Dropout evaluation
        pred_norm = self.model(x, None).squeeze(-1) # Transolver expects (x, None)
        # Unnormalize
        pred_unnorm = self.scaler_U.inverse(pred_norm)
        target_unnorm = self.scaler_U.inverse(y).squeeze(-1)
        # Prepare return values
        coords = self.scaler_X.inverse(x)[..., :self.cfg.problem.model.d] # First d dims are coords
        extras = {}
        return pred_unnorm, target_unnorm, coords, extras
