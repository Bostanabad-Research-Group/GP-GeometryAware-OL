import random
import torch
import numpy as np
from datetime import datetime
import platform
from omegaconf import DictConfig, OmegaConf

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_env_metadata(cfg: DictConfig):
    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "config": OmegaConf.to_container(cfg, resolve=True)
    }
