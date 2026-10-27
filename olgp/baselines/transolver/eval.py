import hydra
from omegaconf import DictConfig

from olgp.baselines.transolver.evaluator import TransolverEvaluator


@hydra.main(version_base=None, config_path="../../../configs", config_name="config_transolver")
def main(cfg: DictConfig):

    evaluator = TransolverEvaluator(cfg)
    evaluator.run()

if __name__ == "__main__":
    main()
