import os
import glob
import numpy as np
import torch
import pyvista as pv
from tqdm import tqdm
from scipy.spatial import KDTree

from olgp.preprocessing.preprocessor import DatasetPreprocessor

class CarDataPreprocessor(DatasetPreprocessor):
    def __init__(self, data_dir_path, out_dir_path, cfg, logger):
        super().__init__("Car", data_dir_path, out_dir_path, cfg, logger)

        self.vtk_files = None
        self.data_points = []
        self.data_pressures = []
        self.data_cds = []
        self.num_points = 3500 # Target number of points per sample

    def load_data(self):
        search_dirs = [f"param{i}" for i in range(9)]
        self.vtk_files = []
        for p_dir in search_dirs:
            full_search = os.path.join(self.data_dir_path, p_dir, "*", "quadpress_smpl.vtk")
            self.logger.info(f"Searching for VTK files in: {full_search}")
            found = glob.glob(full_search)
            self.vtk_files.extend(found)
        self.logger.info(f"Found {len(self.vtk_files)} VTK files.")

    def process_pointcloud(self):
        success_count = 0
        for vtk_file in tqdm(self.vtk_files, desc="Processing"):
            try:
                result = self.process_single_sample(vtk_file)
                if result is not None:
                    pts, prs, cd = result
                    self.data_points.append(pts)
                    self.data_pressures.append(prs)
                    self.data_cds.append(cd)
                    success_count += 1
            except Exception as e:
                self.logger.error(f"Failed to process {vtk_file}: {e}")

        self.logger.info("\n--- Processing Complete ---")
        self.logger.info(f"Successfully processed: {success_count} / {len(self.vtk_files)}")

        self.pointcloud = torch.tensor(np.array(self.data_points), dtype=torch.float32)
        self.output_features = torch.tensor(np.array(self.data_pressures), dtype=torch.float32)
        self.cds = torch.tensor(np.array(self.data_cds), dtype=torch.float32)
        self.logger.info(f"pointcloud shape: {self.pointcloud.shape}, output_features shape: {self.output_features.shape}")
        return self.pointcloud, self.output_features

    def process_global_features(self, step=5):
        return None

    def process_local_features(self):
        return None

    def _resample(self, points, pressure):
        """
        Greedy Farthest Point Sampling (FPS) implementation.
        Guarantees uniform spatial distribution across the car surface.
        """
        N = points.shape[0]

        # Meshes with too few points are padded by repeating points
        if N <= self.num_points:
            idx = np.arange(self.num_points) % N
            return points[idx], pressure[idx]

        # Farthest Point Sampling Algorithm
        farthest_pts = np.zeros(self.num_points, dtype=int)
        distances = np.ones(N) * 1e10

        # Start with 0 point
        farthest_pts[0] = 0

        for i in range(1, self.num_points):
            last_added = points[farthest_pts[i-1]]
            # Calculate squared distance to the last added point
            dist_to_last = np.sum((points - last_added)**2, axis=-1)
            # Update the minimum distances to the current point set
            distances = np.minimum(distances, dist_to_last)
            # Pick the point that is farthest away from all currently selected points
            farthest_pts[i] = np.argmax(distances)

        idx = farthest_pts
        return points[idx], pressure[idx]

    def process_single_sample(self, vtk_path):
        sample_dir = os.path.dirname(vtk_path)

        # --- 1. Load Drag Coefficient File ---
        cd_path = os.path.join(sample_dir, 'cd.txt')
        with open(cd_path, 'r') as f:
            content = f.read().strip().split()
            if not content:
                raise ValueError("Empty cd.txt file")
            cd_val = float(content[0])

        # --- 2. Load VTK & Pressure ---
        mesh = pv.read(vtk_path)
        points_raw = mesh.points
        pressure_raw = np.load(os.path.join(sample_dir, 'press.npy'))

        # --- 3. Isolate Car Surface ---
        bodies = mesh.split_bodies()
        car_body = None
        for body in bodies:
            if (body.bounds[1] - body.bounds[0]) < 8.0 and body.n_points > 200:
                car_body = body
                break

        if car_body is None:
            raise ValueError("Could not identify car body.")

        car_surface = car_body.extract_surface()
        points_clean = np.array(car_surface.points)

        # --- 4. Map Pressures via KDTree ---
        tree = KDTree(points_raw)
        distances, indices = tree.query(points_clean, k=1)
        pressure_clean = pressure_raw[indices]

        # --- 5. Consistent Axis Alignment (Order: 0, 2, 1) ---
        native_x = points_clean[:, 0].copy()
        native_y = points_clean[:, 1].copy()
        native_z = points_clean[:, 2].copy()

        final_points = np.zeros_like(points_clean)
        final_points[:, 0] = native_x  # X: Length
        final_points[:, 1] = native_z  # Y: Width
        final_points[:, 2] = native_y  # Z: Height

        # Centering and Grounding
        final_points[:, 0] -= np.mean(final_points[:, 0])
        final_points[:, 1] -= np.mean(final_points[:, 1])
        final_points[:, 2] -= np.min(final_points[:, 2])

        # --- 6. Resample using Farthest Point Sampling ---
        pts_fixed, press_fixed = self._resample(final_points, pressure_clean)

        return pts_fixed, press_fixed, cd_val
