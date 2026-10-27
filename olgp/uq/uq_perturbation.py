import copy
import warnings
import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset
from olgp.baselines.transolver.evaluator import TransolverEvaluator
from olgp.uq.uq_evaluator import UQEvaluator
from olgp.uq.uq_utils import compute_energy_score_torch
from datetime import datetime

class UQPerturbation(UQEvaluator):
    def __init__(self, cfg, pretrained_no_class, seeds=(0, 1, 2, 3, 4), fit=False):
        """Input perturbation: Gaussian noise on the test inputs."""
        super().__init__(cfg, pretrained_no_class, name="Perturbation", seeds=seeds, fit=fit)
        self.num_perturbations = 50  # perturbed forward passes per seed

    def _fit(self):
        print("This method needs no fit")
        return None, None, False

    def _perturb_test_loader(self, loader, noise_std=10e-3):
        X, _, U = loader.dataset.tensors
        perturbation = torch.randn_like(X) * noise_std
        perturbed_X = X.clone() + perturbation

        new_dataset = TensorDataset(perturbed_X, perturbed_X, U)

        new_loader = DataLoader(
            new_dataset,
            batch_size=loader.batch_size,
            shuffle=False
        )
        return new_loader

    @torch.inference_mode()
    def _evaluate(self):
        pred_mean_list, pred_std_list, Y_test_list, X_test_list = [], [], [], []
        times, memories = [], []
        for i, evaluator in enumerate(self.evaluators):
            start_time = datetime.now()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            evaluator.cfg.settings.random_seed = self.seeds[i]
            evaluator.setup()
            evaluator.model.eval()

            print(f"  Running Test Inference ({self.num_perturbations} passes)...")
            perturbation_runs_preds = []
            clean_loader = copy.deepcopy(evaluator.test_loader)
            evaluator.test_loader = clean_loader
            _, clean_preds, test_targets, test_inputs, _ = evaluator._run_inference_loop()
            for run_idx in tqdm(range(self.num_perturbations), desc=" MC Loop"):
                evaluator.test_loader = self._perturb_test_loader(clean_loader, noise_std=10e-3)
                _, predictions, targets, inputs, _ = evaluator._run_inference_loop()

                perturbation_runs_preds.append(predictions)

            stack = torch.stack(perturbation_runs_preds)  # [passes, N_test, Points]
            seed_es = compute_energy_score_torch(stack, test_targets)
            print(f"Energy Score for Seed {self.seeds[i]}: {seed_es:.4f}")

            if not hasattr(self, 'seed_energy_scores'):
                self.seed_energy_scores = []
            self.seed_energy_scores.append(seed_es)

            model_mean = torch.mean(stack, dim=0)
            model_std = torch.std(stack, dim=0)
            pred_mean_list.append(model_mean.cpu())
            pred_std_list.append(model_std.cpu())
            X_test_list.append(test_inputs.cpu())
            Y_test_list.append(test_targets.cpu())

            end_time = datetime.now()
            times.append((end_time - start_time).total_seconds())
            fit_max_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            memories.append(fit_max_mem_mb)

        # Stack results across seeds
        self.pred_mean = torch.stack(pred_mean_list).squeeze() # [Seeds, Samples, N]
        self.pred_std = torch.stack(pred_std_list).squeeze()
        self.Y_test = torch.stack(Y_test_list)
        self.X_test = torch.stack(X_test_list)

        return times, memories, self.seed_energy_scores

@hydra.main(version_base=None, config_path="../../configs", config_name="config_transolver")
def main(cfg: DictConfig):
    warnings.filterwarnings("ignore", category=UserWarning)
    evaluator = UQPerturbation(cfg, pretrained_no_class=TransolverEvaluator, seeds=[0, 1, 2, 3, 4])
    evaluator.run()

if __name__ == "__main__":
    main()
