# Lambeq for NLI


`./utils/requirements.sh`

# For running domain adaptation experiments snli vs mednli

- confirm config values in `utils/config.py`
- `python trainer_classical_claim_ev_classification.py`
notes to self:
- label mapping of SNLI to float
- neutral=0
- contradiction=1
- entailment=2

# for domain adaptation

# todo as of march 22nd 2023
- Load entire MNLI
- Get training to run
- Debug and find what its doing
- ensure Loss drops accuracy increases
- addin validation
- add in mednli
