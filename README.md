# InterBehav-ZTA: Reproducibility Package

This package contains the data, scripts, and environment specification
required to reproduce every number in the manuscript:

"InterBehav-ZTA: Behavioral Evidence-Driven Zero Trust for Continuous
Adaptive Access Control"
Gauri Korgaonkar, Renuka Pawar, Prasenjit Bhavtankar
Submitted to IEEE Transactions on Dependable and Secure Computing

Version: 1.0
Date: 12/10/2026
Contact: gauri.korgaonkar24@spit.ac.in

---

## 1. Package contents

manuscript/
InterBehav-ZTA_Final.pdf The submitted manuscript

figures/ All figures referenced in the paper

results/ Raw JSON result files

database_exports/ CSV exports of the database tables cited

scripts/ Python scripts used to produce the results

requirements_frozen.txt Exact Python package versions

---

## 2. Environment

The prototype was run on:

Python 3.8.18
Flask 2.2.5
Flask-SQLAlchemy 3.0.5
SQLAlchemy 2.0.22
SQLite 3.x (bundled with Python)
scikit-learn 1.3.2
XGBoost 2.0.3
PyTorch 2.0.1
SHAP 0.42.1
NumPy 1.24.4
pandas 2.0.3

Reference deployment: single Intel-based workstation, 16 GB RAM, no GPU
(CPU inference).

Seed 42 is used for all stochastic splits and detector initialization.

To recreate the environment:

pip install -r requirements_frozen.txt

---

## 3. Result files and their sources

Every table in the manuscript traces to one of the files below.

## Table / Figure Source file Script

Table VIII results/dsl_eer_sweep_results.json scripts/dsl_evaluation.py
Table IX results/balabit_supervised_results.json
Table X results/cbac_crossval_results.json scripts/train_cbac_real.py
Table XI results/dsl_ablation_results.json
Table XII results/dsl_feature_ablation_results.json
Table XIII database_exports/trust_history.csv (queried from live DB)
Table XIV derived from Eq. (24); no data file
Table XV results/policy_simulation_results.json scripts/policy_simulation.py
Fig. 4, 5 results/dsl_eer_sweep_results.json
Fig. 6 results/cbac_crossval_results.json
Fig. 8 results/dsl_ablation_results.json
Fig. 9 analytical; generated from Eq. (20) and (28)
Fig. 10 live DB; see database_exports/trust_history.csv
Fig. 11 live DB; see database_exports/behavioral_trust_scores.csv
Fig. 12 results/policy_simulation_results.json

---

## 4. Database exports

The CSV files under database_exports/ are direct exports of the
following SQLite tables. They back the tables and figures that cite
live deployment counts.

trust_history.csv TrustHistory rows (1,233 in the paper)
cbac_results.csv CBACResult rows
anomaly_results.csv AnomalyResult rows
resource_access_decisions.csv ResourceAccessDecision rows (2 in the paper)
behavioral_trust_scores.csv BehavioralTrustScore rows
experiments.csv Experiment rows
experiment_observations.csv ExperimentObservation rows (labeled windows)

No personal data is included. User identity is preserved only as
the integer user_id, which is the internal database identifier and
carries no personal information on its own. Raw keystroke and mouse
event streams are not exported.

---

## 5. How to reproduce

To regenerate the policy-level evaluation of Section IV-K:

cd backend
python scripts/policy_simulation.py

Output: results/policy_simulation_results.json
Runtime: approximately 30 seconds on CPU.

To regenerate the local multimodal evaluation:

cd backend
python scripts/evaluate_multimodal_final.py

Output: results/multimodal_results.json

To regenerate the database CSV exports from the SQLite file:

cd backend
python scripts/export_csvs.py

Note: the scripts expect the SQLite database to be present at
backend/instance/spit_zero_trust.db. It is not included in this
package because it contains raw telemetry with possible sensitive
content; only the derived CSV exports are provided.

---

## 6. Public datasets

The DSL keystroke benchmark (51 users) is the dataset of Killourhy
and Maxion (DSN 2009), available at:
https://www.cs.cmu.edu/~keystroke/

The Balabit mouse dynamics benchmark (10 users) is available at:
https://github.com/balabit/Mouse-Dynamics-Challenge

Neither dataset is redistributed here; the derived feature vectors
used in the paper are produced by the loader scripts in scripts/.

---

## 7. Notes on the local cohort

The local prototype portal was deployed on 27 enrolled accounts.
Of these:

- 17 accounts have recorded trust trajectories under continuous
  operation (see database_exports/trust_history.csv)
- 4 participants have sufficient labeled windows with complete
  pipeline output for the policy-level replay of Section IV-K
  (see database_exports/experiment_observations.csv)

The 346 labeled windows used in Table XV break down as:

125 NORMAL
134 STRESS
87 MALICIOUS

---

## 8. Known limitations

Section V-I of the manuscript documents the limitations of the
evaluation, including:

- The Balabit mouse features are not discriminative (EER near 47%)
- The local cohort is small (4 participants, 8 sessions)
- The calibration reference in the policy replay includes malicious
  windows, which would not be the case at real enrollment
- Per-window precision and recall are not reported for the live
  telemetry stream because there is no per-window ground truth

These limitations are stated in the paper and reproduced here for
reviewer convenience.

---

## 9. Contact

For questions about this reproducibility package:

Gauri Vinod Korgaonkar
Sardar Patel Institute of Technology
gauri.korgaonkar24@spit.ac.in
