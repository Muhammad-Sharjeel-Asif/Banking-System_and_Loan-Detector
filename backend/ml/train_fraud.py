"""
Fraud Detection Model Training Script

This script trains an Isolation Forest model on the Kaggle Credit Card Fraud
dataset to detect anomalous (potentially fraudulent) transactions.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Author: Sharjeel (as per sharjeel.md assignment)

Usage:
    cd backend
    python ml/train_fraud.py

Note for Mohib (Intern):
    Isolation Forest is an unsupervised anomaly detection algorithm.
    Unlike classification models, it doesn't need labeled fraud/non-fraud data.
    It works by isolating observations - anomalies are easier to isolate,
    so they get shorter path lengths in the decision trees.
    
    This is perfect for fraud detection because:
    1. Fraud is rare (imbalanced data)
    2. New fraud patterns emerge that weren't in training data
    3. We want to flag anything "unusual" not just known fraud patterns
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ── Load Kaggle dataset ───────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Note: creditcard.csv is large (~150MB), so we use a sample for training

# Try to load from current directory first, then ml/ folder
if os.path.exists('creditcard.csv'):
    csv_path = 'creditcard.csv'
elif os.path.exists(os.path.join(os.path.dirname(__file__), 'creditcard.csv')):
    csv_path = os.path.join(os.path.dirname(__file__), 'creditcard.csv')
else:
    print("Error: creditcard.csv not found!")
    print("Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("Place it in the ml/ folder or run this script from the backend/ directory.")
    exit(1)

print("Loading dataset (this may take a moment)...")

# Use a sample for faster training (50,000 records as per sharjeel.md)
df = pd.read_csv(csv_path).sample(n=50000, random_state=42)

print(f"Dataset loaded: {len(df)} records (sampled from full dataset)")

# ── Feature engineering ───────────────────────────────────────────────
# As per sharjeel.md, we use:
# - Amount: Transaction amount in PKR (or USD in Kaggle dataset)
# - Time: Seconds since first transaction (we'll use this for hour detection)

features = ['Amount', 'Time']
X = df[features]

print(f"Features: {features}")
print(f"Amount range: {X['Amount'].min():.2f} - {X['Amount'].max():.2f}")
print(f"Time range: {X['Time'].min():.0f} - {X['Time'].max():.0f} seconds")

# ── Scale features ────────────────────────────────────────────────────
# Isolation Forest works better with scaled features
# StandardScaler transforms data to have mean=0 and std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures scaled (mean=0, std=1)")

# ── Train Isolation Forest model ──────────────────────────────────────
# contamination=0.01: We expect ~1% of data to be anomalies (fraud)
# This is a hyperparameter you can tune based on your data
# random_state=42: Fixed seed for reproducibility

model = IsolationForest(
    contamination=0.01,  # Expected proportion of outliers
    random_state=42,
    n_estimators=100,    # Number of trees (same as RandomForest)
    max_samples='auto'   # Number of samples to draw for each tree
)

model.fit(X_scaled)

print(f"\n=== Model Training Complete ===")
print(f"Model: IsolationForest with {model.n_estimators} trees")

# ── Evaluate on training data (optional) ──────────────────────────────
# Note: Isolation Forest is unsupervised, so we can't use accuracy
# Instead, we check how many samples are flagged as anomalies
predictions = model.predict(X_scaled)
anomaly_count = (predictions == -1).sum()  # -1 = anomaly, 1 = normal

print(f"\nAnomaly Detection Results:")
print(f"  Normal transactions: {(predictions == 1).sum()}")
print(f"  Anomalies detected: {anomaly_count} ({anomaly_count/len(predictions):.2%})")

# If you have the 'Class' column (ground truth), you can compare:
if 'Class' in df.columns:
    actual_fraud = (df['Class'] == 1).sum()
    print(f"  Actual fraud cases in sample: {actual_fraud} ({actual_fraud/len(df):.2%})")

# ── Save model and scaler ─────────────────────────────────────────────
# We need to save both the model AND the scaler
# The scaler is needed to transform new data the same way as training data
output_path = os.path.join(os.path.dirname(__file__), 'fraud_model.pkl')

joblib.dump({
    'model': model,
    'scaler': scaler,
    'features': features
}, output_path)

print(f"\n✓ Model saved to: {output_path}")
print("\nTraining complete!")
print("\nNote for Mohib (Intern):")
print("  The saved model will be loaded by apps/loans/ml_model.py")
print("  It will be used in the detect_fraud() function to flag suspicious transactions.")
