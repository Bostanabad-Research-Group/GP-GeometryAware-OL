import os
import sys
import torch
from pathlib import Path
from datetime import datetime
import logging
import yaml

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from olgp.utils import set_seed
from olgp.gpp.config import configure_logger, logger
from olgp.preprocessing.naca import NacaDataPreprocessor
from olgp.preprocessing.elasticity import ElasticityDataPreprocessor
from olgp.preprocessing.car import CarDataPreprocessor
from olgp.preprocessing.ahmed import AhmedDataPreprocessor
from olgp.preprocessing.pipe import PipeDataPreprocessor
from olgp.utils import get_env_metadata

@hydra.main(version_base=None, config_path="../../../configs", config_name="config_transolver")
def main(cfg: DictConfig):
    #############################################################################################################################
    # CONFIGURATION
    #############################################################################################################################
    ##### Load configuration
    problem     = cfg.problem.name
    random_seed = cfg.settings.random_seed
    dtype       = cfg.settings.dtype
    device      = cfg.settings.device
    base_path   = cfg.machine.base_path
    data_dir    = cfg.workdir.data_dir

    ##### Define Settings
    tkwargs = {
        "dtype": getattr(torch, dtype),
        "device": torch.device(device),
    }
    set_seed(random_seed)

    ##### Define Paths
    # Out path
    output_dir_path = HydraConfig.get().runtime.output_dir
    Path(output_dir_path + "/Benchmarks").mkdir(parents=True, exist_ok=True)
    Path(output_dir_path + "/Plots").mkdir(parents=True, exist_ok=True)
    pre_path = os.path.join(cfg.workdir.base_path, cfg.ablation.preprocess_subdir_train)
    Path(pre_path).mkdir(parents=True, exist_ok=True)
    # Data load path
    data_dir_path = os.path.join(base_path, data_dir, rf'{problem}')

    ##### Configure Logger
    configure_logger(level=logging.DEBUG)
    logger.info(OmegaConf.to_yaml(cfg))

    logger.info("########################################## Start main #################################################")
    start_time = datetime.now()

    #############################################################################################################################
    # PREPROCESSING
    #############################################################################################################################
    ##### Reset before processing
    # Reset Peak Memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    preprocess_start_time = datetime.now()

    ##### SHARED PREPROCESSING STEPS FOR ALL DATASETS #####
    # Select preprocessor
    if problem == "Naca":
        preprocessor = NacaDataPreprocessor(
            data_dir_path=data_dir_path,
            out_dir_path=pre_path,
            cfg=cfg,
            logger=logger
        )
    elif problem == "Elasticity":
        preprocessor = ElasticityDataPreprocessor(
            data_dir_path=data_dir_path,
            out_dir_path=pre_path,
            cfg=cfg,
            logger=logger
        )
    elif problem == "Car":
        preprocessor = CarDataPreprocessor(
            data_dir_path=data_dir_path,
            out_dir_path=pre_path,
            cfg=cfg,
            logger=logger
        )
    elif problem == "Ahmed":
        preprocessor = AhmedDataPreprocessor(
            data_dir_path=data_dir_path,
            out_dir_path=pre_path,
            cfg=cfg,
            logger=logger
        )
    elif problem == "Pipe":
        preprocessor = PipeDataPreprocessor(
            data_dir_path=data_dir_path,
            out_dir_path=pre_path,
            cfg=cfg,
            logger=logger
        )
    else:
        raise ValueError(f"Problem {problem} not supported for preprocessing.")

    preprocessor.preprocess()

    preprocess_end_time = datetime.now()
    preprocess_duration_sec = (preprocess_end_time - preprocess_start_time).total_seconds()
    logger.info(f"Preprocessing duration: {preprocess_duration_sec:.2f} seconds")

    ##### Peak Memory
    peak_memory_gb = 0
    if torch.cuda.is_available():
        # Returns memory in bytes, convert to GB for readability
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    ##### Save benchmarks
    results_path = os.path.join(output_dir_path, "Benchmarks/benchmarks_preprocess.yaml")
    benchmarks = {
        "env": get_env_metadata(cfg),
        "metrics": {
            "peak_memory_gb": peak_memory_gb,
            "total_time_min": preprocess_duration_sec / 60,
        },
    }

    with open(results_path, 'w') as f:
        yaml.dump(benchmarks, f)

    elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
    logger.info(f"[Time] Total Elapsed: {elapsed_minutes:.2f} minutes")
    logger.info("########################################## End main #################################################")

if __name__ == '__main__':
    print("[DEBUG] Start main with sys.argv:", sys.argv)
    main()
