import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from olgp.baselines.base_trainer import BaselineTrainer
from olgp.baselines.transolver.src.testloss import RelativeLpLoss
from olgp.baselines.transolver.src.energyloss import EnergyScoreLoss
from olgp.baselines.transolver.src.model_dict import get_model
from olgp.normalize import StandardScaler

class TransolverTrainer(BaselineTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _load_data_impl(self):
        # Load dataset
        dataset = torch.load(os.path.join(self.pre_path, rf'{self.cfg.problem.name}_data.pt'), weights_only=True)
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
        N_train = self.cfg.problem.data.N_train
        N_val = self.cfg.problem.data.N_val
        N_test = self.cfg.problem.data.N_test
        X_train = X_data[:N_train, ...] # [Nsamples, Npoints, Nfeatures]
        U_train = U_data[:N_train, ...].unsqueeze(-1) # [Nsamples, Npoints, 1]
        self.logger.debug(f"Training Input Shape: {X_train.shape}, Training Output Shape: {U_train.shape}")
        X_val = X_data[-N_test-N_val:-N_test, ...] # [Nsamples, Npoints, Nfeatures]
        U_val = U_data[-N_test-N_val:-N_test, ...].unsqueeze(-1) # [Nsamples, Npoints, 1]
        self.logger.debug(f"Validation Input Shape: {X_val.shape}, Validation Output Shape: {U_val.shape}")
        # Normalize
        scaler_X = StandardScaler()
        scaler_U = StandardScaler()
        X_train_norm, X_mean, X_std = scaler_X.fit_transform(X_train)
        U_train_norm, U_mean, U_std = scaler_U.fit_transform(U_train)
        X_val_norm = scaler_X.transform(X_val)
        U_val_norm = scaler_U.transform(U_val)
        # DataLoaders
        batch_size = self.cfg.problem.training.batch_size
        train_loader = DataLoader(
            TensorDataset(X_train_norm, X_train_norm, U_train_norm), # Transolver passes x_train twice
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val_norm, X_val_norm, U_val_norm),
            batch_size=batch_size,
            shuffle=False,
        )

        # Set final values
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler_X = scaler_X
        self.scaler_U = scaler_U

    def _build_model_impl(self):
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
        self.logger.info(f"Initialized Transolver Model: {self.model}")

    def _setup_optimizer_impl(self):
        """Standard optimizer setup."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.problem.training.lr,
            weight_decay=self.cfg.problem.training.weight_decay
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.cfg.problem.training.lr,
            epochs=self.cfg.problem.training.N_epochs,
            steps_per_epoch=len(self.train_loader),
        )

    def compute_step(self, batch, is_train: bool):
        """Compute forward pass and loss for a batch."""
        # Define loss function
        if self.cfg.ablation.name == "pno":
            self.loss_fn = EnergyScoreLoss(p=2, reduction='mean')
        else:
            self.loss_fn = RelativeLpLoss(size_average=False)
        # Unpack batch
        pos, fx, y = batch
        x, _, y = pos.to(self.device), fx.to(self.device), y.to(self.device)

        #Forward pass
        if self.cfg.ablation.name == "pno":
            M = self.cfg.ablation.M
            preds = torch.stack([self.model(x, None) for _ in range(M)], dim=1)
            pred = preds.mean(dim=1) #[B,N,1]
            loss = self.loss_fn(preds, y)
        else:
            pred = self.model(x, None) #[B,N,1]
            loss = self.loss_fn(pred, y)

        # Backward pass and optimization
        if is_train:
            loss.backward()
            if self.cfg.problem.training.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.problem.training.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

        # Unnormalize predictions and targets
        predictions_unnorm = self.scaler_U.inverse(pred.cpu().detach())
        targets_unnorm = self.scaler_U.inverse(y.cpu().detach())

        return loss.item(), predictions_unnorm, targets_unnorm
