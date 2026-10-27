import os
import numpy as np
import torch

from olgp.preprocessing.preprocessor import DatasetPreprocessor

class PipeDataPreprocessor(DatasetPreprocessor):
    def __init__(self, data_dir_path, out_dir_path, cfg, logger):
        super().__init__("Pipe", data_dir_path, out_dir_path, cfg, logger)

        self.nodes_files = None
        self.features_files = None
        self.num_points = 5000 # Target number of points per sample

        # Lists to hold the processed data
        self.data_points = []
        self.data_pressures = []
        self.data_cds = []

    def load_data(self):
        self.XC = np.load(os.path.join(self.data_dir_path, "Pipe_X.npy"))
        self.YC = np.load(os.path.join(self.data_dir_path, "Pipe_Y.npy"))
        response = np.load(os.path.join(self.data_dir_path, "Pipe_Q.npy"))
        self.QC = np.sqrt(np.power(response[:,0], 2) + np.power(response[:,1], 2))
        self.logger.info(f"XC shape: {self.XC.shape}, YC shape: {self.YC.shape}, QC shape: {self.QC.shape}")

    def process_pointcloud(self):
        X_d = torch.from_numpy(self.XC) # (nsamples, x_coordinates, flow_number) = (2310, 129, 129)
        Y_d = torch.from_numpy(self.YC) # (nsamples, x_coordinates, flow_number) = (2310, 129, 129)
        Q_d = torch.from_numpy(self.QC) # (nsamples, x_coordinates, flow_number) = (2310, 129, 129)
        flows = []
        flows_q = []
        step = 5
        stop_val = X_d.shape[2]
        indices = list(range(0, stop_val, step))
        if indices[-1] != stop_val-1:
            indices.append(stop_val-1)
        for i in indices:
            xy = torch.cat((X_d[:, :, i].unsqueeze(-1), Y_d[:, :, i].unsqueeze(-1)), dim=-1)
            flows.append(xy)
            flows_q.append(Q_d[:,:,i])
        XY_dataset = torch.cat(flows, dim=1) # (nsamples, point_index, xy_coordinates) = (2310, 16641, 2)
        Q_dataset = torch.cat(flows_q, dim=-1) # (nsamples, point_index) = (2310, 16641)
        self.pointcloud = XY_dataset
        self.output_features = Q_dataset
        self.logger.info(f"pointcloud shape: {self.pointcloud.shape}, output_features shape: {self.output_features.shape}")
        return self.pointcloud, self.output_features

    def process_global_features(self, step=5):
        return None

    def process_local_features(self):
        return None
