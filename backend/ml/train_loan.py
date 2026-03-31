"""
Loan Prediction Model Training Script

This script trains a Random Forest classifier on the Kaggle Bank Loan dataset
to predict loan eligibility.

Dataset: https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset

Author: Sharjeel (as per sharjeel.md assignment)

Usage:
    cd backend
    python ml/train_loan.py

Note for Mohib (Intern):
    This script uses scikit-learn's RandomForestClassifier which is an
    ensemble learning method. It builds multiple decision trees and
    outputs the class that is the mode of the classes of the individual trees.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# ── Load Kaggle dataset ───────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset
# Place the CSV file in the ml/ folder or provide full path

# Try to load from current directory first, then ml/ folder
if os.path.exists('loan_data.csv'):
    df = pd.read_csv('loan_data.csv')
elif os.path.exists(os.path.join(os.path.dirname(__file__), 'loan_data.csv')):
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'loan_data.csv'))
else:
    print("Error: loan_data.csv not found!")
    print("Please download from: https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset")
    print("Place it in the ml/ folder or run this script from the backend/ directory.")
    exit(1)

# ── Clean data ────────────────────────────────────────────────────────
# Drop rows with missing values (simple approach for this project)
df = df.dropna()

print(f"Dataset loaded: {len(df)} records after cleaning")

# ── Feature engineering ───────────────────────────────────────────────
# Map our banking system features to Kaggle dataset columns:
# - ApplicantIncome → yearly_balance (user's income from transactions)
# - CoapplicantIncome → 0 (we don't track co-applicants)
# - LoanAmount → loan amount requested
# - Loan_Amount_Term → default 360 months
# - Credit_History → has_collateral (1 or 0)

# Select features as specified in sharjeel.md
features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History']

# Fill any remaining NaN values with 0
X = df[features].fillna(0)

# Target variable: Loan_Status ('Y' = 1, 'N' = 0)
y = (df['Loan_Status'] == 'Y').astype(int)

print(f"Features: {features}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# ── Train/Test split ──────────────────────────────────────────────────
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42  # Fixed seed for reproducibility
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ── Train Random Forest model ─────────────────────────────────────────
# n_estimators=100: Number of trees in the forest
# random_state=42: Fixed seed for reproducibility
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate model ────────────────────────────────────────────────────
accuracy = model.score(X_test, y_test)
print(f"\n=== Model Performance ===")
print(f"Accuracy: {accuracy:.2%}")

# Feature importance (useful for explaining to teacher)
print("\nFeature Importance:")
for feature, importance in zip(features, model.feature_importances_):
    print(f"  {feature}: {importance:.2%}")

# ── Save model ────────────────────────────────────────────────────────
# Save to ml/ folder where Django will load it from
output_path = os.path.join(os.path.dirname(__file__), 'loan_model.pkl')
joblib.dump(model, output_path)

print(f"\n✓ Model saved to: {output_path}")
print("Training complete!")