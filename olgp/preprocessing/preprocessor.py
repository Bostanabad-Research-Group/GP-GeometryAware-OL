from abc import ABC, abstractmethod
import os

import torch

class DatasetPreprocessor(ABC):
    def __init__(self, problem, data_dir_path, out_dir_path, cfg, logger):
        self.problem = problem
        self.data_dir_path = data_dir_path
        self.out_dir_path = out_dir_path
        self.cfg = cfg
        self.logger = logger

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def process_pointcloud(self):
        pass

    @abstractmethod
    def process_global_features(self):
        pass

    @abstractmethod
    def process_local_features(self):
        pass

    def preprocess(self):
        # 0. Load dataset
        self.load_data()
        # 1. Process pointcloud
        pointcloud, output_features = self.process_pointcloud()
        # 2. Process global features
        global_features = self.process_global_features()
        # 3. Process local features
        local_features = self.process_local_features()

        try:
            torch.save({
                "pointcloud": pointcloud,
                "output_features": output_features,
                "global_features": global_features,
                "local_features": local_features,
            }, os.path.join(self.out_dir_path, rf"{self.problem}_data.pt"))
        except Exception as e:
            self.logger.error(f"Error saving preprocessed data: {e}")
            raise e

