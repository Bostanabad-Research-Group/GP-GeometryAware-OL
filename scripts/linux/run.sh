#!/bin/bash
export PYTHONPATH="<ROOT_FOLDER>:$PYTHONPATH"

# REEF-GP
python <ROOT_FOLDER>/reefgp/olgp/uq/uq_ours.py --config-name config_transolver problem.uq.fit=True problem.uq.aggregation="gpoe" problem="transolver/elasticity" "$@"
python <ROOT_FOLDER>/reefgp/olgp/uq/uq_ours.py --config-name config_transolver problem.uq.fit=True problem.uq.aggregation="gpoe" problem="transolver/naca" "$@"
python <ROOT_FOLDER>/reefgp/olgp/uq/uq_ours.py --config-name config_transolver problem.uq.fit=True problem.uq.aggregation="gpoe" problem="transolver/pipe" "$@"
python <ROOT_FOLDER>/reefgp/olgp/uq/uq_ours.py --config-name config_transolver problem.uq.fit=True problem.uq.aggregation="gpoe" problem="transolver/car" "$@"
python <ROOT_FOLDER>/reefgp/olgp/uq/uq_ours.py --config-name config_transolver problem.uq.fit=True problem.uq.aggregation="gpoe" problem="transolver/ahmed" "$@"