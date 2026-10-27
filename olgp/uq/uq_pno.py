import warnings
import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from olgp.baselines.transolver.evaluator import TransolverEvaluator
from olgp.uq.uq_evaluator import UQEvaluator
from olgp.uq.uq_utils import compute_energy_score_torch
from datetime import datetime

class UQPNO(UQEvaluator):
    def __init__(self, cfg, pretrained_no_class, seeds=(0, 1, 2, 3, 4), fit=False):
        """Probabilistic neural operator: dropout samples from the backbone trained with the energy score."""
        super().__init__(cfg, pretrained_no_class, name="PNO", seeds=seeds, fit=fit)
        self.mc_passes = 50  # stochastic forward passes per seed

        if cfg.ablation.name != "pno":
            raise ValueError(f"Config mismatch: Expected ablation.name to be 'pno', but got '{cfg.ablation.name}'")
        if cfg.problem.model.dropout == 0.0:
            raise ValueError(f"Config mismatch: Expected problem.model.dropout to be > 0 for PNO, but got {cfg.problem.model.dropout}")

    def _fit(self):
        print("This method needs no fit")
        return None, None, False

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
            evaluator.setup(set_fixed_seed=False)
            evaluator.model.eval()

            print(f"  Running Test Inference ({self.mc_passes} passes)...")
            mc_dropout_runs_preds = []
            test_targets = None
            test_inputs = None
            for run_idx in tqdm(range(self.mc_passes), desc="  MC Loop"):
                # The evaluator keeps dropout active for the dropout and pno ablations
                _, predictions, targets, inputs, _ = evaluator._run_inference_loop()

                mc_dropout_runs_preds.append(predictions)
                if test_targets is None:
                    test_targets = targets
                    test_inputs = inputs

            stack = torch.stack(mc_dropout_runs_preds)  # [passes, N_test, Points]
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
    evaluator = UQPNO(cfg, pretrained_no_class=TransolverEvaluator, seeds=[0, 1, 2, 3, 4])
    evaluator.run()

if __name__ == "__main__":
    main()
