import hydra
from omegaconf import DictConfig

from olgp.baselines.transolver.trainer import TransolverTrainer


@hydra.main(version_base=None, config_path="../../../configs", config_name="config_transolver")
def main(cfg: DictConfig):

    trainer = TransolverTrainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()
