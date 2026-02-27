# Credit Card Fraud Detection

##  Overview

This project builds and evaluates machine learning models to detect fraudulent credit card transactions from a highly imbalanced dataset (0.17% fraud rate).  

Rather than focusing on accuracy, the analysis emphasizes:
- Precision–recall trade-offs  
- Threshold tuning  
- Cross-validation  
- Business-aware model selection  

---

##  Dataset

- 284,807 transactions  
- 492 fraud cases (~0.17%)  
- 30 numerical features (V1–V28 are PCA-transformed), plus `Time` and `Amount`  
- Target variable: `Class` (0 = Normal, 1 = Fraud)

The dataset is highly imbalanced, making precision and recall more meaningful than accuracy.

---

## Key Findings from EDA

- Fraud cases are extremely rare.
- Transaction amount alone is not a reliable fraud indicator.
- PCA features such as **V14, V12, V17, and V10** show strong class separation.
- Several features exhibit strong negative correlation with fraud.

---

## Modeling Strategy

### 1. Baseline – Logistic Regression
- Precision: 0.84  
- Recall: 0.68  
- ROC-AUC: 0.948  

High precision but missed many fraud cases.

---

### 2. Imbalance Handling
Applied:
- `class_weight='balanced'`
- Feature scaling
- Stratified train-test split

Result:
- Recall improved to 0.92  
- Precision dropped significantly (too many false positives)

---

### 3. Threshold Optimization

Using the Precision–Recall curve, a tuned probability threshold (~0.9999) was selected.

Final tuned model:
- Precision: **0.75**
- Recall: **0.84**
- F1-score: 0.79
- ROC-AUC: 0.972

This reduced missed fraud cases by ~48% compared to baseline while keeping false positives manageable.

---

## Cross-Validation (5-Fold Stratified)

| Model | Mean ROC-AUC |
|--------|--------------|
| Logistic Regression | **0.979** |
| Random Forest | 0.949 |

Logistic Regression showed higher performance and better stability across folds.

---

## Final Model Comparison (Test Set)

| Model | Precision | Recall | F1 | ROC-AUC |
|--------|-----------|--------|------|---------|
| Logistic (Baseline) | 0.84 | 0.68 | 0.75 | 0.948 |
| Logistic (Balanced 0.5) | 0.06 | 0.92 | 0.11 | 0.972 |
| Logistic (Tuned) | 0.75 | 0.84 | 0.79 | 0.972 |
| Random Forest | 0.96 | 0.74 | 0.84 | 0.953 |

Selected Model: **Tuned Logistic Regression**

Reason: Best trade-off between fraud detection and operational false alarms.

---

##  Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Stratified K-Fold Cross-Validation

---

## Key Takeaways

- Accuracy is misleading in imbalanced datasets.
- Precision–recall analysis is critical for fraud detection.
- Threshold tuning can significantly improve business outcomes.
- Cross-validation ensures model stability.

---
