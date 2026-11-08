# Stroke Risk Prediction on Imbalanced Clinical Data

Predicting stroke risk when only **4.87%** of patients had a stroke — and showing why accuracy is a trap.

A model with 95% accuracy can still miss 49 out of 50 strokes. This repo proves that, then fixes it with threshold tuning and resampling — and shows where SMOTE helps, where it doesn't, and how leakage can fake a 99% score.

> ⚠️ Not a medical device. This is a modelling study on a public Kaggle dataset. It must not be used for clinical decisions.

## The problem

- **Dataset:** Healthcare Dataset Stroke Data (Kaggle), `healthcare-dataset-stroke-data.csv`
- **Size:** 5,110 patients, 12 columns. Stroke = 249 (4.87%), No stroke = 4,861 (95.13%)
- **Split:** stratified 80/20 — train 4,087 (199 strokes), test 1,023 (50 strokes). `SEED = 4`
- **Missing:** BMI missing in 201 rows, imputed with the median
- **Features:** age, hypertension, heart disease, avg glucose, BMI, smoking status, work type, residence, marital status, gender (1 `Other` row dropped)

## The accuracy trap

| Model (default threshold 0.5) | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC | Missed strokes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy (always "no stroke") | 0.9511 | — | 0.00 | 0.00 | — | — | 50 / 50 |
| Logistic Regression | 0.9521 | 1.00 | **0.02** | 0.0392 | 0.8394 | 0.2692 | **49 / 50** |

Both look ~95% accurate. The logistic model catches 1 stroke in 50. In a hospital, a false negative means sending a high-risk patient home — so recall matters more than accuracy here.

## What actually worked

### 1. Threshold tuning (best simple fix)

Tuned on out-of-fold train probabilities only, then applied once to test:

| Approach | Threshold | Precision | Recall | F1 | False alarms | Missed strokes |
|---|---:|---:|---:|---:|---:|---:|
| LR, default | 0.50 | 1.0000 | 0.02 | 0.0392 | 0 | 49 |
| LR, tuned (OOF) | 0.0509 | 0.1349 | 0.82 | — | 263 | 9 |
| LR, chosen | 0.056 | 0.1418 | **0.80** | 0.2410 | 242 | **10** |

Trade-off, stated plainly: catching 40 of 50 strokes costs ~242 false alarms out of 1,023 patients. For screening that trade is defensible; for diagnosis it is not.

### 2. Resampling comparison (test set)

| Model + method | Precision | Recall | F1 | ROC AUC | PR AUC |
|---|---:|---:|---:|---:|---:|
| LR, no resampling | 1.0000 | 0.02 | 0.0392 | 0.8394 | 0.2692 |
| LR, class_weight=balanced | 0.1347 | 0.80 | 0.2305 | 0.8395 | 0.2595 |
| LR, SMOTE | 0.1338 | 0.80 | 0.2292 | **0.8406** | **0.2876** |
| LR, RandomUnderSampler | 0.1246 | 0.82 | 0.2164 | 0.8352 | 0.2338 |
| RF, class_weight=balanced | 0.1895 | 0.72 | **0.3000** | 0.8150 | 0.1987 |
| RF, SMOTE | 0.5000 | 0.06 | 0.1071 | 0.7977 | 0.2290 |

SMOTE helps Logistic Regression slightly on PR AUC, but it hurts Random Forest badly (recall 0.06). No single resampling method wins everywhere — that is the finding.

### 3. Bias–variance

Decision Tree validation curve (5-fold stratified CV, ROC AUC): best `max_depth = 4` (CV AUC 0.8180, train AUC 0.8580). Depth `None`: train AUC 1.0000, CV AUC 0.5644 — textbook overfitting. Bagging 40 bootstrapped trees lifts the depth-3 tree from single-tree AUC 0.7964 to averaged AUC 0.8340.

### 4. Leakage demo — the most important table here

| SMOTE timing | Precision | Recall | F1 | ROC AUC | PR AUC |
|---|---:|---:|---:|---:|---:|
| Correct: SMOTE **after** split (train only) | 0.5000 | 0.0600 | 0.1071 | 0.7977 | 0.2290 |
| Wrong: SMOTE **before** split | 0.9957 | 0.9475 | 0.9710 | 0.9955 | 0.9961 |

SMOTE before the split leaks test information into training and fakes a near-perfect score. If you see 99% on imbalanced data, check the split first.

### 5. Noise robustness

Adding Gaussian noise to age / glucose / BMI: Logistic Regression ROC AUC falls 0.8394 (no noise) → 0.7834 (noise level 0.20). Random Forest falls faster (0.8224 → 0.7076). With 10% flipped labels, SMOTE does not rescue Random Forest (ROC AUC 0.7637 → 0.6811) while Logistic Regression is roughly flat (0.7985 → 0.7992).

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Lab9_Bias_Variance_Imbalance_Template.ipynb
```

Get the dataset from Kaggle (Healthcare Dataset Stroke Data) and place `healthcare-dataset-stroke-data.csv` next to the notebook — the file is not committed to this repo.

## Structure

```
.
├── Lab9_Bias_Variance_Imbalance_Template.ipynb  # full analysis, Parts 0–7
├── README.md
└── requirements.txt
```

## Limitations

- Single dataset, single split (80/20, seed 4) — no nested CV for the final threshold.
- Precision is low (~0.14 at recall 0.80): this is a screening-style trade-off, not a diagnostic model.
- No calibration, no fairness audit by age/gender, no external validation on another hospital's data.
- Next step: turn the tuned Logistic Regression into a FastAPI service (`stroke-risk-ml-api`) with validation, explainability, and tests — see the roadmap in the notebook's final section.

## Author

Bahadur Ali Meer Mohammad — BS Computer Science (AI), Sukkur IBA University.
Machine Learning Engineer / Agent Builder (LLM Engineer). LinkedIn: https://www.linkedin.com/in/bahadur-ali82/
