import warnings
import torch
import hydra
from omegaconf import DictConfig
from olgp.utils import set_seed
from olgp.baselines.transolver.evaluator import TransolverEvaluator
from olgp.uq.uq_evaluator import UQEvaluator
from olgp.uq.uq_utils import compute_energy_score_torch
from datetime import datetime
import numpy as np

class UQEnsemble(UQEvaluator):
    def __init__(self, cfg, pretrained_no_class, seeds=(0, 1, 2, 3, 4), fit=False):
        """Deep ensemble of the backbones trained with different seeds."""
        super().__init__(cfg, pretrained_no_class, name="Ensemble", seeds=seeds, fit=fit)

    def _fit(self):
        print("This method needs no fit")
        return 0, 0, False

    @torch.inference_mode()
    def _evaluate(self):
        ensemble_predictions = []
        start_time = datetime.now()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        for i, evaluator in enumerate(self.evaluators):

            print(f"\n--- Evaluator for Seed {self.seeds[i]} ---")
            evaluator.cfg.settings.random_seed = self.seeds[i]
            set_seed(self.seeds[i])
            evaluator.setup()
            metrics, predictions, targets, inputs, extras = evaluator._run_inference_loop()
            ensemble_predictions.append(predictions)

        # Ensemble statistics across seeds
        stack = torch.stack(ensemble_predictions).transpose(-1, -2).squeeze(-1)  # [N_seeds, N_test, Points]

        ensemble_es = compute_energy_score_torch(stack, targets)
        print(f"Energy Score for the Deep Ensemble: {ensemble_es:.4f}")
        self.seed_energy_scores = [ensemble_es]  # a single ensemble, so a single score

        pred_mean = torch.mean(stack, dim=0)
        pred_std = torch.std(stack, dim=0)

        # Targets and inputs are the same for every seed
        self.X_test = torch.stack([inputs]).cpu()
        self.Y_test = torch.stack([targets]).cpu()
        self.pred_mean = torch.stack([pred_mean]).cpu()
        self.pred_std = torch.stack([pred_std]).cpu()

        end_time = datetime.now()
        times = np.array((end_time - start_time).total_seconds())
        fit_max_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        memories = np.array([fit_max_mem_mb])

        return times, memories, self.seed_energy_scores

@hydra.main(version_base=None, config_path="../../configs", config_name="config_transolver")
def main(cfg: DictConfig):
    warnings.filterwarnings("ignore", category=UserWarning)
    evaluator = UQEnsemble(cfg, pretrained_no_class=TransolverEvaluator, seeds=[0, 1, 2, 3, 4])
    evaluator.run()

if __name__ == "__main__":
    main()
