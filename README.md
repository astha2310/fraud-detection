# Credit Card Fraud Detection — ML on Real Transaction Data

Built by Astha Patel | M.S. Information Technology, Arizona State University

---

## Why I Built This

Fraud detection is one of the most real-world applications of machine learning. Financial institutions lose billions annually to credit card fraud, and the models that catch it have to work with extreme class imbalance and almost zero tolerance for missing actual fraud.

I wanted to understand how these models actually work, so I built one from scratch on a real dataset of 284,807 credit card transactions.

---

## The Dataset

284,807 real credit card transactions from European cardholders in September 2013. Only 492 were fraudulent, which is 0.173% of the total. Features are PCA-transformed (V1 through V28) to protect cardholder privacy, plus transaction time and amount.

The 0.173% fraud rate is the core challenge. A model that predicts "not fraud" every time would be 99.83% accurate and completely useless. Getting to something genuinely useful requires rethinking how you evaluate performance entirely.

---

## Results

| Model | AUC-ROC | Precision | Recall |
|---|---|---|---|
| Random Forest | 97.66% | 81.82% | 82.65% |
| Logistic Regression | 94.2% | 74.3% | 78.1% |

The Random Forest model identified $60,127 in fraud losses in the test set, catching 82.65% of all fraudulent transactions.

---

## Tech Stack

Python 3, pandas, NumPy, scikit-learn, matplotlib, seaborn, imbalanced-learn

---

## How to Run It

```bash
git clone https://github.com/astha2310/fraud-detection.git
cd fraud-detection

pip3 install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn

# Download creditcard.csv from Kaggle and place in data/ folder
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

python3 fraud_analysis.py
```

---

## Visualizations

The project generates 7 visualizations covering class distribution, transaction amount analysis, time-based fraud patterns, feature correlation heatmap, confusion matrices, ROC curves, and feature importance rankings.

---

## What I Learned

The biggest lesson was that class imbalance changes everything. Every choice from the evaluation metric to the sampling strategy to the classification threshold has to account for it.

The most interesting finding: fraud is not hiding in large transactions. I expected the highest-value transactions to be the most targeted. They are not. Fraud concentrates in small, repeated transactions that look normal individually. That finding completely changed how I thought about the model.

I also found that transaction timing is more predictive than expected. Fraudulent transactions cluster at different hours than legitimate ones, likely because fraud happens when cardholders are less likely to notice.

---

## What Could Be Added Next

Real-time scoring pipeline using a streaming framework, SHAP values for explainability on each prediction, drift detection to catch when transaction patterns shift over time, and integration with a case management system for fraud analysts.

---

Astha Patel | github.com/astha2310 | linkedin.com/in/asthap23
