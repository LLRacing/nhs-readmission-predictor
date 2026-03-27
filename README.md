# NHS Hospital Readmission Predictor
![Python](https://img.shields.io/badge/python-3.11-blue)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

> Predicting 30-day hospital readmission risk using patient admission data, built as part of an MSc in Artificial Intelligence at the University of Liverpool.

---

## Problem Statement

**What problem are we solving?**
Unplanned 30-day hospital readmissions cost the NHS approximately £2.4bn annually. When a patient is discharged and returns within 30 days as an emergency, it often signals that the underlying condition was not fully resolved — or that the patient lacked adequate support at home.

Currently, identifying high-risk patients before discharge relies heavily on clinical intuition. This is inconsistent — a busy ward may miss warning signs that a more systematic approach would catch.

**Why ML?**
A machine learning model can systematically score every patient at discharge, flagging high-risk individuals for targeted interventions such as GP follow-up calls, community nursing referrals, or enhanced discharge planning. Unlike manual review, it scales to every patient on every ward, every day.

> Having supported NHS infrastructure at Stockport NHS Foundation Trust and Manchester University NHS Foundation Trust, I wanted to explore whether ML could help identify high-risk patients before discharge — and whether the findings could inform real discharge planning workflows.

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | [MIMIC-IV Clinical Database](https://physionet.org/content/mimiciv/) (PhysioNet) |
| Institution | Beth Israel Deaconess Medical Center, Boston, USA |
| Size | 500,000+ admissions across 200,000+ patients (demo: 275 admissions, 100 patients) |
| Time period | 2008–2022 (shifted forward for de-identification) |
| Access | Requires credentialed PhysioNet account, CITI training, and signed Data Use Agreement |
| Tables used | `admissions.csv`, `patients.csv`, `diagnoses_icd.csv`, `procedures_icd.csv` |

**How to get the data:**
1. Register at [physionet.org](https://physionet.org)
2. Complete the CITI "Data or Specimens Only Research" course at [citiprogram.org](https://citiprogram.org)
3. Submit credentialing details and training certificate on PhysioNet
4. Sign the Data Use Agreement on the MIMIC-IV project page
5. Download CSV files to `data/raw/` (approval typically takes 24–48 hours)

A publicly available demo dataset (100 patients) is available without registration:
[MIMIC-IV Clinical Database Demo](https://physionet.org/content/mimic-iv-demo/)

---

## Approach / Methodology

The project follows the standard ML workflow:

1. **Exploratory Data Analysis** — understand admission patterns, data quality, and class distribution
2. **Data Cleaning** — handle missing discharge locations, exclude in-hospital deaths, fix data types
3. **Feature Engineering** — calculate length of stay, prior admissions, emergency flag, and Charlson Comorbidity Index from ICD codes
4. **Baseline Model** — Logistic Regression as interpretable benchmark
5. **Model Improvement** — Random Forest and XGBoost with hyperparameter tuning
6. **Evaluation** — ROC-AUC, Precision, Recall, F1 with focus on Recall (missing a high-risk patient is more costly than a false alarm)
7. **Explainability** — SHAP values to identify key clinical drivers of readmission risk

---

## Features

| Feature | Type | Description | Why included |
|---------|------|-------------|--------------|
| `anchor_age` | Numeric | Patient age at reference year | Older patients have higher readmission risk |
| `gender` | Categorical | Patient gender (M/F) | Clinical literature shows gender differences in readmission |
| `is_emergency` | Binary | 1 if emergency admission, 0 otherwise | Emergency admissions have significantly higher readmission rates |
| `length_of_stay` | Numeric | Days between admission and discharge | Longer stays may indicate more complex conditions |
| `prior_admissions_12m` | Numeric | Number of admissions in prior 12 months | Strongest predictor of future readmission |
| `discharge_location` | Categorical | Where patient went after discharge | Patients discharged home alone are at higher risk |
| `num_diagnoses` | Numeric | Number of ICD diagnosis codes | Proxy for clinical complexity |
| `num_procedures` | Numeric | Number of procedures during stay | Proxy for severity of intervention |

**Target variable:** `readmitted_30d` — 1 if patient was admitted again within 30 days of discharge, 0 otherwise. Patients who died in hospital are excluded.

---

## Models Trained

| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | Baseline | Simple, interpretable benchmark |
| Random Forest | Ensemble | Handles non-linearity and feature interactions |
| XGBoost | Gradient boosting | Best performance, tuned with GridSearchCV |

---

## Results

| Model | ROC-AUC | Precision | Recall | F1 Score |
|-------|---------|-----------|--------|----------|
| Logistic Regression | 0.XX | 0.XX | 0.XX | 0.XX |
| Random Forest | 0.XX | 0.XX | 0.XX | 0.XX |
| XGBoost | 0.XX | 0.XX | 0.XX | 0.XX |

**Best model:** [To be completed] 

> 📊 See `results/figures/` for ROC curves, confusion matrix, and SHAP summary plot.

---

## Key Findings

> *(To be completed after model training)*

- Top predictors of readmission (from SHAP analysis): [TBC]
- Emergency admissions vs elective readmission rate comparison: [TBC]
- Impact of discharge location on readmission risk: [TBC]
- Performance on high-risk subgroups (elderly, multiple prior admissions): [TBC]

---

## Data Notes & Limitations

- **American vs NHS data:** MIMIC-IV is sourced from a US hospital. Clinical patterns are broadly similar to NHS data, but admission type labels, coding systems, and patient demographics differ. The model would require revalidation on NHS trust data before any clinical use.
- **Date de-identification:** All dates are shifted randomly into the future per patient for privacy. Relative time between events (e.g. days between admissions) is preserved and accurate.
- **Patient age approximation:** `anchor_age` represents the patient's age at `anchor_year`, not their exact age at each admission. Used as a reasonable approximation.
- **Missing discharge locations:** 42 rows (demo dataset) had missing `discharge_location`. Where `hospital_expire_flag = 1`, filled with `'DIED'`. Remainder filled with `'Unknown'` to preserve row count.
- **In-hospital deaths excluded:** 15 patients who died during admission were removed from readmission analysis — deceased patients cannot be readmitted.
- **Class imbalance:** Readmission is a relatively rare event (~15–20% of admissions). This imbalance will be addressed during model training.
- **Not a clinical tool:** This model is an academic portfolio project and has not been validated for clinical use. It should not be used to make or influence patient care decisions without appropriate clinical governance review.

---

## Project Structure

```
nhs-readmission-predictor/
│
├── data/
│   ├── raw/                  # Original MIMIC-IV CSV files (not included — see Dataset)
│   └── processed/            # Cleaned, feature-engineered data
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_features.ipynb     # Feature engineering
│   └── 03_modelling.ipynb    # Model training and evaluation
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Load and validate raw MIMIC-IV data
│   ├── features.py           # Feature engineering functions
│   ├── train.py              # Train and evaluate models
│   └── explain.py            # SHAP explainability analysis
│
├── models/                   # Saved model artefacts (.pkl)
├── results/
│   └── figures/              # ROC curves, SHAP plots, confusion matrix
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/LLRacing/nhs-readmission-predictor.git
cd nhs-readmission-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add MIMIC-IV CSV files to data/raw/
#    (requires PhysioNet credentialed access — see Dataset section)
#    Or use the demo dataset for testing (no registration required)

# 4. Run notebooks in order
jupyter notebook notebooks/01_eda.ipynb

# 5. Or run the full training pipeline
python src/train.py
```

---

## Real-World Application

This project was developed with potential NHS deployment in mind, drawing on direct experience working at Stockport NHS Foundation Trust and Manchester University NHS Foundation Trust.

A realistic phased deployment approach would be:

**Phase 1 — Retrospective analysis** *(low risk, no governance barriers)*
Use anonymised trust discharge data to replicate the analysis on local patient populations. Output: a report of top readmission risk factors specific to the trust.

**Phase 2 — Pilot as decision-support tool** *(requires Information Governance approval)*
Surface readmission risk scores to discharge coordinators for high-risk patients only. Clinicians retain full decision-making authority — the model supports, not replaces, clinical judgement.

**Phase 3 — Full governance review** *(required before clinical use)*
Any clinical deployment would require review under NHS Information Governance, Caldicott Guardian approval, and MHRA AI as a Medical Device framework assessment.

---

## Future Work

- [ ] Train and evaluate on full MIMIC-IV dataset (200,000+ patients)
- [ ] Add Charlson Comorbidity Index from ICD-10 diagnosis codes
- [ ] Build a Streamlit dashboard for interactive risk scoring
- [ ] Validate model performance on NHS open data
- [ ] Explore fairness metrics across age, gender, and ethnicity subgroups

---

## Requirements

```
pandas==2.2.0
numpy==1.26.4
scikit-learn==1.4.0
xgboost==2.0.3
shap==0.44.0
matplotlib==3.8.2
seaborn==0.13.2
jupyter==1.0.0
```

Install all with: `pip install -r requirements.txt`

---

## Author

**Larry Lee**
- MSc Artificial Intelligence, University of Liverpool (2025–present)
- Senior IT Infrastructure Engineer, Stockport NHS Foundation Trust
- [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
- [GitHub](https://github.com/LLRacing)

---

## License

This project is licensed under the MIT License.
MIMIC-IV data is subject to PhysioNet's Data Use Agreement and must not be redistributed.

---

## Acknowledgements

- Johnson, A., et al. MIMIC-IV (version 3.1). PhysioNet. 2024.
- [PhysioNet](https://physionet.org) for providing access to the MIMIC-IV dataset
- University of Liverpool MSc AI programme
