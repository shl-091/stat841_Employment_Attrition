# Employee Attrition Prediction

Employee attrition is a critical challenge for organizations, leading to increased costs and loss of institutional knowledge.  
This project applies machine learning classification techniques to predict employee attrition using the **IBM HR Analytics Employee Attrition** dataset.

The goal is to:

- Identify **at-risk employees**  
- Understand **key drivers of turnover** (e.g., overtime, income, job role)  
- Support **proactive HR interventions** rather than reactive responses  

---

## 👥 Contributors

- **Judy Li**  
  - Model implementation: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting  
  - Hyperparameter tuning  
  - SMOTE implementation and optimization  

- **Zihan Huang**  
  - Exploratory Data Analysis (EDA)  
  - Feature engineering  
  - Data cleaning  
  - Business insight generation  

---

## 📊 Dataset

- **Source:** Kaggle – *IBM HR Analytics Employee Attrition*  
- **Size:** 1,470 observations, 35 variables  
- **Target Variable:**  
  - `Attrition` (Binary: `Yes` / `No`)  

- **Key Features (examples):**
  - `Age`  
  - `MonthlyIncome`  
  - `OverTime`  
  - `JobRole`  
  - `YearsWithCurrManager`  
  - and other demographic, job-related, and performance-related features  

---

## ⚙️ Methodology

### 1. Data Preprocessing

- **Cleaning**
  - Removed redundant or constant columns  
    e.g., `EmployeeCount`, `Over18`, etc.
- **Encoding**
  - Applied **One-Hot Encoding** for categorical variables.
- **Scaling**
  - Used **StandardScaler** for numerical features to standardize input for certain models.
- **Feature Selection**
  - Selected **top 10 features** based on:
    - Correlation with attrition  
    - Attrition rate differences across feature categories (attrition spread)

---

### 2. Handling Class Imbalance

- The original dataset is **imbalanced**:
  - ~16% of employees labeled as `Attrition = Yes`
- To address this:
  - Applied **SMOTE (Synthetic Minority Over-sampling Technique)** **only on the training set**  
  - This balances the classes and improves the model’s ability to detect attrition (higher recall).

---

### 3. Models Implemented

The following classification models were trained and evaluated:

1. **Logistic Regression** (Baseline)  
2. **Decision Tree**  
3. **Random Forest**  
4. **Gradient Boosting** (Best overall performer)

All models were evaluated on a held-out test set using multiple classification metrics.

---

## 📈 Model Performance

All metrics are reported on the **test set**.

| Model             | Accuracy | Precision | Recall | F1-Score |  AUC  |
|------------------|:--------:|:---------:|:------:|:--------:|:-----:|
| Logistic Regression | 72.1% | 31.6% | 63.8% | 0.423 | 0.733 |
| Decision Tree       | 78.6% | 36.7% | 46.8% | 0.411 | 0.657 |
| Random Forest       | 86.1% | 59.4% | 40.4% | 0.481 | 0.740 |
| Gradient Boosting   | 84.4% | 51.3% | 42.6% | 0.465 | 0.749 |

> **Note on formatting:**  
> - Accuracy and precision/recall are shown as **percentages** (one decimal place).  
> - F1-Score and AUC are shown as **decimal values** in the range [0, 1].

---

## 🎯 Threshold Optimization (Early Warning System)

While accuracy is important, HR use cases often prioritize **catching as many at-risk employees as possible** (high recall), even at the cost of some additional false positives.

- The default decision threshold is **0.5** (predict “Yes” if `P(Attrition) ≥ 0.5`).
- For the **Gradient Boosting** model:
  - We lowered the threshold from **0.5 → 0.25**.
  - This adjustment:
    - **Increased Recall** to **61.7%**  
    - Still maintained **better precision** than the Logistic Regression baseline.

This makes the model more suitable as an **“early warning system”**, enabling HR to:

- Flag more employees who might leave  
- Prioritize follow-up actions and retention strategies  

---

## 🔍 Key Drivers of Attrition

Based on feature importance from tree-based models and exploratory analysis, key drivers include:

1. **OverTime**  
   - Strongest predictor of attrition; employees frequently working overtime are at higher risk.

2. **Job Role**  
   - Certain roles, particularly **Sales Representatives**, exhibit a higher attrition rate.

3. **Monthly Income**  
   - Lower income levels are associated with higher attrition probability.

4. **Years With Current Manager**  
   - Shorter tenure with the current manager is linked to higher attrition, possibly indicating adjustment or management issues.

These insights can guide HR in designing **targeted interventions**, such as:

- Monitoring overtime policies  
- Reviewing compensation for high-risk roles  
- Supporting new manager–employee relationships  

---

## 💻 Installation & Usage

### 1. Dependencies

This project requires **Python 3.8+** and the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
