import os

import numpy as np
import torch

from olgp.preprocessing.preprocessor import DatasetPreprocessor

class NacaDataPreprocessor(DatasetPreprocessor):
    def __init__(self, data_dir_path, out_dir_path, cfg, logger):
        super().__init__("Naca", data_dir_path, out_dir_path, cfg, logger)

        self.nx = 40  # crop angle (from 0 - counterclockwise)   - Angle control
        self.n = 20   # number of radial layers                  - Radial control
        self.XC = None
        self.YC = None
        self.QC = None

    def load_data(self):
        self.XC = np.load(os.path.join(self.data_dir_path, "NACA_Cylinder_X.npy"))
        self.YC = np.load(os.path.join(self.data_dir_path, "NACA_Cylinder_Y.npy"))
        self.QC = np.load(os.path.join(self.data_dir_path, "NACA_Cylinder_Q.npy"))
        self.logger.info(f"XC shape: {self.XC.shape}, YC shape: {self.YC.shape}, QC shape: {self.QC.shape}")

    def process_pointcloud(self):
        nx, n = self.nx, self.n
        XC_dataset = torch.from_numpy(self.XC[:,nx:-nx,0:n]).float()
        YC_dataset = torch.from_numpy(self.YC[:,nx:-nx,0:n]).float()
        QC_dataset = torch.from_numpy(self.QC[:,4, nx:-nx,0:n]).float()

        XC_dataset = XC_dataset.flatten(1,2)
        YC_dataset = YC_dataset.flatten(1,2)
        self.logger.debug(f"XC_dataset shape: {XC_dataset.shape}, YC_dataset shape: {YC_dataset.shape}")

        pointcloud = torch.stack([XC_dataset, YC_dataset], dim=-1)
        output_features = QC_dataset.flatten(1,2)
        self.pointcloud = pointcloud
        self.output_features = output_features
        self.logger.info(f"pointcloud shape: {pointcloud.shape}, output_features shape: {output_features.shape}")
        return pointcloud, output_features

    def process_global_features(self, step=5):
        global_features = None
        self.logger.info(f"global_features shape: {None}")
        return global_features

    def process_local_features(self):
        local_features = None
        self.logger.info(f"Local features shape: {None}")
        return local_features
