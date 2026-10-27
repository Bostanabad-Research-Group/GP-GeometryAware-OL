#!/bin/bash
export PYTHONPATH="<ROOT_FOLDER>:$PYTHONPATH"

python -c "import torch; torch.cuda.empty_cache()"
python <ROOT_FOLDER>/olgp/baselines/transolver/preprocess.py --config-name config_transolver "$@"