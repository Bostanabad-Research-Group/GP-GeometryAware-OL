#!/bin/bash
export PYTHONPATH="<ROOT_FOLDER>:$PYTHONPATH"

# Elasticity
python <ROOT_FOLDER>/reefgp/olgp/uq/uq_ensemble.py --config-name config_transolver problem.uq.fit=True problem="transolver/elasticity" "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_mcdropout.py --config-name config_transolver problem.uq.fit=True  problem="transolver/elasticity" ablation=dropout problem.model.dropout=0.1 "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_pno.py --config-name config_transolver problem.uq.fit=True problem="transolver/elasticity" ablation=pno problem.model.dropout=0.1 "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_la.py --config-name config_transolver problem.uq.fit=True problem="transolver/elasticity" "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_perturbation.py --config-name config_transolver problem.uq.fit=True problem="transolver/elasticity" "$@"

# Naca
python <ROOT_FOLDER>/reefgp/olgp/uq/uq_ensemble.py --config-name config_transolver problem.uq.fit=True problem="transolver/naca" "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_mcdropout.py --config-name config_transolver problem.uq.fit=True  problem="transolver/naca" ablation=dropout problem.model.dropout=0.1 "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_pno.py --config-name config_transolver problem.uq.fit=True problem="transolver/naca" ablation=pno problem.model.dropout=0.1 "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_la.py --config-name config_transolver problem.uq.fit=True problem="transolver/naca" "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_perturbation.py --config-name config_transolver problem.uq.fit=True problem="transolver/naca" "$@"

# Pipe
python <ROOT_FOLDER>/reefgp/olgp/uq/uq_ensemble.py --config-name config_transolver problem.uq.fit=True problem="transolver/pipe" "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_mcdropout.py --config-name config_transolver problem.uq.fit=True   problem="transolver/pipe" ablation=dropout problem.model.dropout=0.1 "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_pno.py --config-name config_transolver problem.uq.fit=True problem="transolver/pipe" ablation=pno problem.model.dropout=0.1 "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_la.py --config-name config_transolver problem.uq.fit=True problem="transolver/pipe" "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_perturbation.py --config-name config_transolver problem.uq.fit=True problem="transolver/pipe" "$@"

# Car
python <ROOT_FOLDER>/reefgp/olgp/uq/uq_ensemble.py --config-name config_transolver problem.uq.fit=True problem="transolver/car" "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_mcdropout.py --config-name config_transolver problem.uq.fit=True   problem="transolver/car" ablation=dropout problem.model.dropout=0.1 "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_pno.py --config-name config_transolver problem.uq.fit=True problem="transolver/car" ablation=pno problem.model.dropout=0.1 "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_la.py --config-name config_transolver problem.uq.fit=True problem="transolver/car" "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_perturbation.py --config-name config_transolver problem.uq.fit=True problem="transolver/car" "$@"

# Ahmed
python <ROOT_FOLDER>/reefgp/olgp/uq/uq_ensemble.py --config-name config_transolver problem.uq.fit=True problem="transolver/ahmed" "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_mcdropout.py --config-name config_transolver problem.uq.fit=True   problem="transolver/ahmed" ablation=dropout problem.model.dropout=0.1 "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_pno.py --config-name config_transolver problem.uq.fit=True problem="transolver/ahmed" ablation=pno problem.model.dropout=0.1 "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_la.py --config-name config_transolver problem.uq.fit=True problem="transolver/ahmed" "$@"
# python <ROOT_FOLDER>/reefgp/olgp/uq/uq_perturbation.py --config-name config_transolver problem.uq.fit=True problem="transolver/ahmed" "$@"
