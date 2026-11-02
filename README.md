# USANS milk gelation analysis

This repository contains a compact, reproducible data and code package for the manuscript:

**From microstructure to elasticity: time-resolved USANS reveals length-scale-dependent gelation pathways and structure-rheology associations in milk acid gels**

The study reanalyses time-resolved ultra-small-angle neutron scattering (USANS) and rheology data for cow, goat, and sheep milk before and after heating. No new measurements were made. The analysis accounts for sequential USANS acquisition by assigning each common q feature its estimated acquisition time.

## Scope

The public package provides:

- the authoritative run-to-condition mapping used in the revision;
- the synchronized 16-channel USANS-rheology dataset used for correlation and modelling;
- q-wise structural midpoint estimates and uncertainty summaries;
- reference outputs for structure-rheology correlations and nested leave-one-system-out validation;
- scripts that reproduce the central descriptive and validation summaries.

Raw beamline files, manuscript drafts, reviewer correspondence, and internal audit files are not included.

## Repository structure

```text
data/
  authoritative_run_to_condition.csv
  q_synchronised_USANS_rheology_dataset.csv
results/
  kinetic_summary_reference.csv
  nested_LOSO_model_metrics_reference.csv
  qwise_t50_reference.csv
  structure_rheology_correlations_reference.csv
scripts/
  reproduce_core_results.py
  reproduce_loso_models.py
```

## Reproduce the summaries

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/reproduce_core_results.py
python scripts/reproduce_loso_models.py
```

Generated files are written to `reproduced_results/`.

The random seed is fixed at `20260924`. Preprocessing is fitted on training data only. The strict validation script uses six outer folds, each holding out one complete milk system; hyperparameters are selected by an inner leave-one-system-out loop over the five remaining systems.

## Data notes

- `sample_id` is the stable experimental-series identifier.
- `time_min` is measured from the recorded start of the first USANS scan for each system.
- `rheology_time_min` is the time coordinate used to align rheology with USANS.
- `logI_q_*` columns are synchronized log10 USANS intensities at the 16 common q values.
- `log10_Gprime` is log10 storage modulus in Pa.
- `GH_rpt`/KKB0018333 is unheated goat and `G_rpt`/KKB0018343 is heated goat, following the experimenter-confirmed mapping. Beamline names alone were not used to infer goat heat treatment.

## Interpretation limits

One selected USANS acidification run was analysed per condition. Repeated scans are longitudinal observations, not independent preparations; between-run variability therefore cannot be estimated. Uncertainty intervals derived from archived intensity errors and acquisition-time brackets are conditional within-run quantities, not biological confidence intervals. Whole-pattern convergence is an operational distance to the final observed profile, not proof of equilibrium. Machine-learning results are exploratory and should not be treated as deployment-ready prediction.

## Source study

The measurements were originally reported in Yang et al. (2023), *Food Hydrocolloids*, 137, 108381. https://doi.org/10.1016/j.foodhyd.2022.108381

## License

Code is released under the MIT License. Processed data are released under CC BY 4.0; please cite the manuscript and the source study when reusing them.
