import os

import torch

from olgp.preprocessing.preprocessor import DatasetPreprocessor

class ElasticityDataPreprocessor(DatasetPreprocessor):
    def __init__(self, data_dir_path, out_dir_path, cfg, logger):
        super().__init__("Elasticity", data_dir_path, out_dir_path, cfg, logger)

        self.dataset = None

    def load_data(self):
        self.dataset = torch.load(os.path.join(self.data_dir_path, "Elasticity.pt"), weights_only=True)

    def process_pointcloud(self):
        pointcloud = self.dataset['original_input']
        output_features = self.dataset['original_output']
        self.logger.info(f"pointcloud shape: {pointcloud.shape}, output_features shape: {output_features.shape}")
        return pointcloud, output_features

    def process_global_features(self):
        T = self.dataset['T']
        global_features = T
        self.logger.info(f"global_features shape: {global_features.shape}")
        return global_features

    def process_local_features(self):
        local_features = None
        self.logger.info(f"Local features shape: {None}")
        return local_features


