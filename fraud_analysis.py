"""
Credit Card Fraud Detection & Analysis
By Astha Patel | Data Analyst

A comprehensive analysis of 284,807 real credit card transactions
to identify fraud patterns and build a machine learning detection model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              average_precision_score)
import warnings
warnings.filterwarnings('ignore')

# Set visual style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print("   CREDIT CARD FRAUD DETECTION & ANALYSIS")
print("   By Astha Patel | Data Analyst")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1] Loading dataset...")
df = pd.read_csv('data/creditcard.csv')
print(f"    Total transactions: {len(df):,}")
print(f"    Total features: {df.shape[1]}")
print(f"    Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n[2] Exploratory Data Analysis...")

# Class distribution
fraud_count = df['Class'].sum()
legit_count = len(df) - fraud_count
fraud_pct = (fraud_count / len(df)) * 100

print(f"    Legitimate transactions: {legit_count:,} ({100-fraud_pct:.3f}%)")
print(f"    Fraudulent transactions: {fraud_count:,} ({fraud_pct:.3f}%)")
print(f"    Class imbalance ratio: 1 fraud per {legit_count // fraud_count:,} legitimate")

# Missing values
print(f"    Missing values: {df.isnull().sum().sum()}")

# Amount statistics
print(f"\n    Transaction amount statistics:")
print(f"    - Min: ${df['Amount'].min():.2f}")
print(f"    - Max: ${df['Amount'].max():,.2f}")
print(f"    - Mean: ${df['Amount'].mean():.2f}")
print(f"    - Median: ${df['Amount'].median():.2f}")

print(f"\n    Fraud vs Legitimate amounts:")
print(f"    - Avg fraud amount: ${df[df['Class']==1]['Amount'].mean():.2f}")
print(f"    - Avg legit amount: ${df[df['Class']==0]['Amount'].mean():.2f}")

# ============================================================
# 3. VISUALIZATIONS
# ============================================================
print("\n[3] Generating visualizations...")

# --- VIZ 1: Class Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
counts = df['Class'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].bar(['Legitimate', 'Fraud'], counts.values, color=colors, edgecolor='black')
axes[0].set_title('Transaction Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Transactions')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5000, f'{v:,}', ha='center', fontweight='bold')

# Pie chart with percentages
axes[1].pie(counts.values, labels=['Legitimate', 'Fraud'], colors=colors,
            autopct='%1.3f%%', startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Class Distribution (%)', fontsize=14, fontweight='bold')

plt.suptitle('Credit Card Fraud: Highly Imbalanced Dataset', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('visualizations/01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("    [+] 01_class_distribution.png")

# --- VIZ 2: Transaction Amount Analysis ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Amount distribution by class
legit_amounts = df[df['Class'] == 0]['Amount']
fraud_amounts = df[df['Class'] == 1]['Amount']

axes[0].hist(legit_amounts[legit_amounts < 500], bins=50, alpha=0.6,
             label='Legitimate', color='#2ecc71', edgecolor='black')
axes[0].hist(fraud_amounts[fraud_amounts < 500], bins=50, alpha=0.6,
             label='Fraud', color='#e74c3c', edgecolor='black')
axes[0].set_xlabel('Transaction Amount ($)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Transaction Amount Distribution (< $500)', fontweight='bold')
axes[0].legend()

# Boxplot comparison
data_box = [df[df['Class']==0]['Amount'], df[df['Class']==1]['Amount']]
box = axes[1].boxplot(data_box, labels=['Legitimate', 'Fraud'], patch_artist=True)
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
axes[1].set_ylabel('Amount ($)')
axes[1].set_title('Amount Distribution by Class (Boxplot)', fontweight='bold')
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('visualizations/02_amount_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("    [+] 02_amount_analysis.png")

# --- VIZ 3: Time Pattern Analysis ---
df['Hour'] = (df['Time'] // 3600) % 24

fig, ax = plt.subplots(figsize=(14, 5))
fraud_by_hour = df[df['Class']==1].groupby('Hour').size()
legit_by_hour = df[df['Class']==0].groupby('Hour').size()

# Normalize to compare patterns
fraud_pct_by_hour = (fraud_by_hour / fraud_by_hour.sum()) * 100
legit_pct_by_hour = (legit_by_hour / legit_by_hour.sum()) * 100

x = np.arange(24)
width = 0.35
ax.bar(x - width/2, legit_pct_by_hour, width, label='Legitimate', color='#2ecc71', edgecolor='black')
ax.bar(x + width/2, fraud_pct_by_hour, width, label='Fraud', color='#e74c3c', edgecolor='black')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('% of Transactions')
ax.set_title('Transaction Patterns by Hour: Fraud vs Legitimate', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.legend()
plt.tight_layout()
plt.savefig('visualizations/03_time_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print("    [+] 03_time_patterns.png")

# --- VIZ 4: Correlation Heatmap (Top correlated features with Class) ---
correlations = df.corr()['Class'].abs().sort_values(ascending=False)
top_features = correlations[1:11].index.tolist()  # Top 10 excluding Class itself

fig, ax = plt.subplots(figsize=(10, 6))
top_corr = df[top_features + ['Class']].corr()
sns.heatmap(top_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Top 10 Features Correlated with Fraud', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/04_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("    [+] 04_correlation_heatmap.png")

# ============================================================
# 4. MACHINE LEARNING MODEL
# ============================================================
print("\n[4] Building fraud detection ML model...")

# Prepare features
X = df.drop(['Class', 'Hour'], axis=1)
y = df['Class']

# Scale Amount and Time
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time'] = scaler.fit_transform(X[['Time']])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"    Training set: {len(X_train):,} transactions")
print(f"    Test set: {len(X_test):,} transactions")

# Train Logistic Regression
print("\n    Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_probs = lr_model.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)

# Train Random Forest
print("    Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                   random_state=42, n_jobs=-1, max_depth=10)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)

# Results
print(f"\n    [+] Logistic Regression AUC-ROC: {lr_auc:.4f}")
print(f"    [+] Random Forest AUC-ROC: {rf_auc:.4f}")

# --- VIZ 5: Confusion Matrix ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_lr = confusion_matrix(y_test, lr_preds)
cm_rf = confusion_matrix(y_test, rf_preds)

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
axes[0].set_title(f'Logistic Regression\nAUC-ROC: {lr_auc:.4f}', fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
axes[1].set_title(f'Random Forest\nAUC-ROC: {rf_auc:.4f}', fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('visualizations/05_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("    [+] 05_confusion_matrices.png")

# --- VIZ 6: ROC Curve ---
fig, ax = plt.subplots(figsize=(10, 6))

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

ax.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.4f})',
        linewidth=2, color='#3498db')
ax.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.4f})',
        linewidth=2, color='#2ecc71')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves: Fraud Detection Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/06_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("    [+] 06_roc_curves.png")

# --- VIZ 7: Feature Importance ---
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importance['feature'][::-1], feature_importance['importance'][::-1],
        color='#9b59b6', edgecolor='black')
ax.set_xlabel('Feature Importance')
ax.set_title('Top 15 Most Important Features for Fraud Detection',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/07_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("    [+] 07_feature_importance.png")

# ============================================================
# 5. KEY FINDINGS REPORT
# ============================================================
print("\n[5] Key Findings:")

# Calculate business impact
fraud_amount = df[df['Class']==1]['Amount'].sum()
total_amount = df['Amount'].sum()
fraud_amount_pct = (fraud_amount / total_amount) * 100

print(f"\n    KEY METRICS:")
print(f"    - Dataset spans: 2 days of European cardholder data")
print(f"    - Total fraud loss: ${fraud_amount:,.2f}")
print(f"    - Fraud as % of volume: {fraud_amount_pct:.3f}%")
print(f"    - Best model (Random Forest) AUC-ROC: {rf_auc:.4f}")

# Recall calculation
tn, fp, fn, tp = cm_rf.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"    - Random Forest Precision: {precision:.2%}")
print(f"    - Random Forest Recall: {recall:.2%}")
print(f"    - True frauds caught: {tp}/{tp+fn}")

# Save findings to JSON
findings = {
    'total_transactions': int(len(df)),
    'fraud_transactions': int(fraud_count),
    'legitimate_transactions': int(legit_count),
    'fraud_percentage': round(fraud_pct, 4),
    'total_fraud_amount': round(fraud_amount, 2),
    'avg_fraud_amount': round(df[df['Class']==1]['Amount'].mean(), 2),
    'avg_legit_amount': round(df[df['Class']==0]['Amount'].mean(), 2),
    'logistic_regression_auc': round(lr_auc, 4),
    'random_forest_auc': round(rf_auc, 4),
    'random_forest_precision': round(precision, 4),
    'random_forest_recall': round(recall, 4),
    'frauds_detected': int(tp),
    'frauds_total': int(tp + fn)
}

import json
with open('findings.json', 'w') as f:
    json.dump(findings, f, indent=2)
print("\n    [+] Saved findings.json")

print("\n" + "=" * 70)
print("[+] Analysis complete!")
print(f"[+] {len([f for f in __import__('os').listdir('visualizations') if f.endswith('.png')])} visualizations generated")
print("=" * 70)
