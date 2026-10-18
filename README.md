# CodeAlpha_CreditScoring

**Credit Scoring Model** | CodeAlpha Machine Learning Internship

A machine learning model that predicts whether a loan applicant is **creditworthy**
(likely to repay) or **risky** (likely to default), based on their financial history.

---

## Problem Statement

Banks need to decide who gets a loan. Approving a risky applicant causes financial
loss; rejecting a good applicant loses business. This project frames that decision
as a **binary classification** problem and compares three algorithms to find the one
that separates the two groups best.

---

## Dataset

**UCI German Credit Data** (`credit-g`), loaded automatically from OpenML.

| | |
|---|---|
| Customers | 1,000 |
| Features after encoding | 50 |
| Missing values | 0 |
| Class balance | 70% good / 30% risky |
| Target | `creditworthy` — 1 = good customer, 0 = risky |

The script also contains a **synthetic data generator** as a fallback, so it still
runs end-to-end when there is no internet connection.

---

## Approach

1. **Data loading** — real UCI dataset, with an automatic offline fallback.
2. **Feature engineering** — two new features created from existing columns:
   - `credit_per_month` = credit amount ÷ loan duration → the monthly repayment burden
   - `amount_per_age` = credit amount ÷ age → loan size relative to life stage
3. **Encoding** — categorical columns converted to numbers with one-hot encoding.
4. **Splitting** — 80% train / 20% test, stratified to keep the class ratio equal.
5. **Class imbalance** — `class_weight="balanced"` used so the minority (risky) class
   is not ignored by the models.
6. **Modelling** — Logistic Regression (scaled via Pipeline), Decision Tree, Random Forest.
7. **Validation** — 5-fold stratified cross-validation, so the score is not the result
   of one lucky split.
8. **Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC.

---

## Results

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1-Score | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7788 | 0.675 | 0.8319 | 0.6714 | 0.7431 | 0.7558 |
| Decision Tree | 0.7304 | 0.650 | 0.8571 | 0.6000 | 0.7059 | 0.7354 |
| **Random Forest** ⭐ | **0.8022** | **0.700** | 0.8125 | **0.7429** | **0.7761** | **0.7855** |

**Best model: Random Forest (ROC-AUC 0.7855)**

Confusion matrix for the best model (200 test customers):

| | Predicted Risky | Predicted Good |
|---|---|---|
| **Actually Risky** | 36 | 24 |
| **Actually Good** | 36 | 104 |

Out of 60 genuinely risky applicants, the model correctly flagged 36. The remaining
24 are **false approvals**, which is the costliest error type for a bank and the
clearest target for future improvement.

### Most important features

1. `checking_status_no checking` — applicants with no checking account carry the highest risk
2. `credit_per_month` — *engineered feature*
3. `amount_per_age` — *engineered feature*
4. `credit_amount`
5. `duration`
6. `age`

Both engineered features ranked in the top three, which shows that feature
engineering measurably improved the model.

---

## Files

```
task1_credit_scoring.py             # complete, commented source code
task1_credit_scoring_results.png    # confusion matrix, ROC curves, feature importance
requirements.txt                    # dependencies
README.md
```

## Setup and Run

```bash
pip install -r requirements.txt
python task1_credit_scoring.py
```

The script prints every step to the console and saves the results chart as a PNG.
It also ends with a live prediction for a new applicant (approve / reject + confidence).

---

## Key Takeaways

- Random Forest outperformed the linear model, which suggests the relationship between
  financial history and default risk is **non-linear**.
- Accuracy alone is misleading on imbalanced data — always predicting "good" would score
  70% while catching zero risky applicants. **Recall and ROC-AUC** are the meaningful metrics here.
- ROC-AUC around 0.78 is the normal range for this dataset; real human behaviour is
  inherently less predictable than clean synthetic data.

### Possible improvements
- Threshold tuning to reduce false approvals (the expensive error)
- Hyperparameter search with GridSearchCV
- Gradient boosting models (XGBoost, LightGBM)
- SHAP values so each rejection can be explained to the customer

---

## Author
Tanvir Rahman Anik,
Machine Learning Intern @ CodeAlpha
