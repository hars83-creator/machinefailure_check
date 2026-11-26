# Machine Failure Prediction API - Lab 11

Complete implementation of Lab 11 using the AI4I 2020 Predictive Maintenance dataset.

## Problem
Predict whether a machine will fail on its next run from product type and five sensor/usage measurements, package the fitted preprocessing+model pipelines, and serve predictions through FastAPI.

## Inputs and target
Inputs: `type` (L/M/H), `air_temp` (K), `process_temp` (K), `rpm`, `torque` (Nm), and `tool_wear` (min). Target: `failure` (0/1). `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are removed; the failure-type flags would leak the answer.

## Method
80/20 stratified split (`random_state=42`). Numeric features use `StandardScaler`; `type` uses `OneHotEncoder(handle_unknown="ignore")`. Logistic Regression and Random Forest use balanced class weights. XGBoost uses the negative/positive training ratio as `scale_pos_weight`. Every saved `.joblib` contains the complete preprocessing + classifier pipeline.

## Actual offline results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8245 | 0.1418 | 0.8235 | 0.2419 | 0.9070 | 0.3818 | 12 | 339 |
| Random Forest | 0.9810 | 0.9167 | 0.4853 | 0.6346 | 0.9634 | 0.7768 | 35 | 3 |
| XGBoost | 0.9790 | 0.6585 | 0.7941 | 0.7200 | 0.9628 | 0.8203 | 14 | 28 |

Exact timings and model sizes from the latest run are stored in `model_comparison.csv` because timings vary by machine/run.

## API
Start with `uvicorn app:app --reload --port 8000`.

- `GET /health`
- `POST /predict/logreg`
- `POST /predict/rf`
- `POST /predict/xgb`
- `POST /compare`

Example JSON:
```json
{"type":"L","air_temp":302.0,"process_temp":310.0,"rpm":1300,"torque":65.0,"tool_wear":220}
```

`test_api.py` executes health, healthy/risky comparisons, invalid type (422), missing torque (422), negative rpm (422), unknown model (404), five genuine held-out rows, and 100 requests per model for actual API latency.

## Threshold experiment - XGBoost
At threshold 0.5: recall 0.7941, precision 0.6585, F1 0.7200, FN 14, FP 28. At 0.3: recall rises to 0.8088 and FN falls to 13, but precision falls to 0.5140, F1 falls to 0.6286, and FP rises to 52. Threshold 0.3 catches one additional failure but nearly doubles false alarms; 0.5 is the better default unless the plant establishes that one avoided missed failure is worth the extra inspections.

## Lab questions
1. Recall ranking: Logistic Regression > XGBoost > Random Forest. Precision ranking: Random Forest > XGBoost > Logistic Regression. They differ because the models trade sensitivity for false alarms differently.
2. A false negative is usually more costly because a failing machine is missed. Recall should therefore be a major decision metric, considered with precision/PR-AUC to avoid unusable alert volume.
3. Logistic Regression's low F1 and PR-AUC show that a simple linear decision boundary does not separate failure cases well enough under the class imbalance; nonlinear interactions are useful.
4. Random Forest is much larger because it stores 300 decision trees and their nodes. Size matters for constrained devices, transfer/storage, startup time, and deployment updates.
5. Model probabilities can differ because the algorithms learn different boundaries/calibration. If two say fail and one says OK, use the validated deployment model/threshold and operational risk policy rather than majority vote unless an ensemble was separately validated.
6. For XGBoost, lowering 0.5 to 0.3 changes recall 0.7941 -> 0.8088, FN 14 -> 13, FP 28 -> 52, precision 0.6585 -> 0.5140, and F1 0.7200 -> 0.6286. I would keep 0.5 by default and lower it only if plant costs justify the extra false alarms.
7. Deploy XGBoost: it has the best F1 (0.7200) and PR-AUC (0.8203), strong recall (0.7941), and only 14 false negatives while avoiding Logistic Regression's 339 false positives.
8. Models are loaded once at startup so every request avoids disk I/O/deserialization overhead, reducing latency and resource use.

## Setup and run
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn app:app --reload --port 8000
```
In a second activated terminal:
```bash
python test_api.py
```
Swagger UI: `http://127.0.0.1:8000/docs`.
