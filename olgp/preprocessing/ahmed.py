import os
import glob
import numpy as np
import torch
from tqdm import tqdm

from olgp.preprocessing.preprocessor import DatasetPreprocessor

class AhmedDataPreprocessor(DatasetPreprocessor):
    def __init__(self, data_dir_path, out_dir_path, cfg, logger):
        super().__init__("Ahmed", data_dir_path, out_dir_path, cfg, logger)

        self.nodes_files = None
        self.features_files = None
        self.num_points = 5000 # Target number of points per sample

        # Lists to hold the processed data
        self.data_points = []
        self.data_pressures = []
        self.data_cds = []

    def load_data(self):
        # Your specific file pattern
        self.nodes_files = sorted(glob.glob(os.path.join(self.data_dir_path, 'nodes_*.npy')))
        self.features_files = sorted(glob.glob(os.path.join(self.data_dir_path, 'features_*.npy')))

        if len(self.nodes_files) != len(self.features_files):
            self.logger.warning(f"Mismatch: Found {len(self.nodes_files)} node files and {len(self.features_files)} feature files.")

        self.logger.info(f"Found {len(self.nodes_files)} samples.")

    def process_pointcloud(self):
        success_count = 0
        # Iterate using index to ensure nodes and features match
        for i in tqdm(range(len(self.nodes_files)), desc="Processing Ahmed Bodies"):
            try:
                node_path = self.nodes_files[i]
                feature_path = self.features_files[i]

                result = self.process_single_sample(node_path, feature_path)

                if result is not None:
                    pts, prs = result
                    self.data_points.append(pts)
                    self.data_pressures.append(prs)
                    success_count += 1
            except Exception as e:
                self.logger.error(f"Failed to process sample {i}: {e}")

        self.logger.info(f"\nSuccessfully processed: {success_count} / {len(self.nodes_files)}")

        # Convert to Tensors
        self.pointcloud = torch.tensor(np.array(self.data_points), dtype=torch.float32)
        self.output_features = torch.tensor(np.array(self.data_pressures), dtype=torch.float32)

        # Rearrange
        self.pointcloud[:, :, 0] = -self.pointcloud[:, :, 0] # Negate x
        self.pointcloud[:, :, 1] = -self.pointcloud[:, :, 1] # Negate y

        self.logger.info(f"Pointcloud shape: {self.pointcloud.shape}, Output features shape: {self.output_features.shape}")
        return self.pointcloud, self.output_features

    def process_global_features(self, step=5):
        # Extract inlet velocity and reynolds number
        global_feats_list = []
        for i in range(len(self.nodes_files)):
            feature_path = self.features_files[i]
            features_all = np.load(feature_path)
            global_feats = features_all[:, -2:]
            # Compare all rows to the value in the first row [0]
            feat_2_consistent = (features_all[:, -2] == features_all[0, -2]).all()
            feat_1_consistent = (features_all[:, -1] == features_all[0, -1]).all()
            all_consistent = feat_2_consistent and feat_1_consistent
            print(f"Sample {i}: Inlet Velocity Consistent: {feat_2_consistent}, Reynolds Number Consistent: {feat_1_consistent}, All Consistent: {all_consistent}")
            global_feats_list.append(global_feats[0]) # Assuming all points have the same global features
        global_features = torch.tensor(global_feats_list)
        self.logger.info(f"Global features shape: {global_features.shape}")
        self.global_features = global_features
        return global_features

    def process_local_features(self):
        return None

    def _resample(self, points, pressure):
        N = points.shape[0]
        M = self.num_points

        if N <= M:
            idx = np.arange(M) % N
            return points[idx], pressure[idx]

        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pts_tensor = torch.tensor(points, device=device)

        farthest_pts = torch.zeros(M, dtype=torch.long, device=device)
        distances = torch.full((N,), float('inf'), device=device)

        farthest_pts[0] = 0

        for i in range(1, M):
            last_added_pt = pts_tensor[farthest_pts[i-1]]

            # Distance calculation on GPU
            dist_sq = torch.sum((pts_tensor - last_added_pt)**2, dim=-1)
            distances = torch.minimum(distances, dist_sq)
            farthest_pts[i] = torch.argmax(distances)

        # Move back to CPU and convert to numpy array
        idx = farthest_pts.cpu().numpy()
        return points[idx], pressure[idx]

    def process_single_sample(self, node_path, feature_path):
        # 1. Load the raw numpy arrays
        points_clean = np.load(node_path)       # Shape: (N, 3)
        features_all = np.load(feature_path)   # Shape: (N, num_features)

        # 2. Extract only Pressure (Index 0)
        pressure_clean = features_all[:, 0]

        # 4. Resample using Farthest Point Sampling
        pts_fixed, press_fixed = self._resample(points_clean, pressure_clean)

        return pts_fixed, press_fixed
