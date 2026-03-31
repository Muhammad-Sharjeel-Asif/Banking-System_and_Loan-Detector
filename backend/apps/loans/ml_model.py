"""
ML Model Module for Loan Prediction and Fraud Detection

This module loads trained ML models and provides prediction functions.
Loaded once when Django starts for efficiency.

Author: Sharjeel (as per sharjeel.md assignment)
"""

import joblib
import os
import pandas as pd
import numpy as np
import datetime

# ── Paths ──────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../ml')
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
LOAN_MODEL_PATH = os.path.join(MODEL_DIR, 'loan_model.pkl')
FRAUD_MODEL_PATH = os.path.join(MODEL_DIR, 'fraud_model.pkl')
FRAUD_CSV = os.path.join(DATA_DIR, 'fraud_flags.csv')
STMTS_DIR = os.path.join(DATA_DIR, 'statements')

# ── Load Loan Model ────────────────────────────────────────────────────
# Load model once when Django starts for efficiency
loan_model = None
if os.path.exists(LOAN_MODEL_PATH):
    loan_model = joblib.load(LOAN_MODEL_PATH)
    print("Loan prediction model loaded successfully!")
else:
    print("Warning: loan_model.pkl not found. Run ml/train_loan.py first.")

# ── Load Fraud Model ───────────────────────────────────────────────────
fraud_artifacts = None
if os.path.exists(FRAUD_MODEL_PATH):
    fraud_artifacts = joblib.load(FRAUD_MODEL_PATH)
    print("Fraud detection model loaded successfully!")
else:
    print("Warning: fraud_model.pkl not found. Run ml/train_fraud.py first.")


# ══════════════════════════════════════════════════════════════════════
#  LOAN PREDICTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def compute_user_features(user_id, amount, has_collateral):
    """
    Read user's statement CSV and compute features for ML loan model.
    
    This function maps our banking system features to the Kaggle dataset
    features used during training.
    
    Args:
        user_id: The user's ID to read their statement CSV
        amount: The loan amount being requested
        has_collateral: Boolean indicating if user has collateral
        
    Returns:
        Dictionary with features matching the training data format:
        - ApplicantIncome: Estimated yearly income from credits
        - CoapplicantIncome: Always 0 (we don't track co-applicants)
        - LoanAmount: Loan amount in thousands (matching Kaggle format)
        - Loan_Amount_Term: Default 360 months
        - Credit_History: 1 if has collateral or good income, else 0
    
    Note for Mohib (Intern):
        This function handles the case where a user has no transaction
        history (new account) by returning default values.
    """
    stmt_path = os.path.join(STMTS_DIR, f'user_{user_id}.csv')
    
    try:
        df = pd.read_csv(stmt_path)
        
        # Handle empty statement (new user with no transactions)
        if df.empty:
            return {
                'ApplicantIncome': 0,
                'CoapplicantIncome': 0,
                'LoanAmount': amount / 1000,  # Kaggle uses thousands
                'Loan_Amount_Term': 360,
                'Credit_History': 1 if has_collateral else 0
            }
        
        # Calculate yearly income from credit transactions
        credits = df[df['type'] == 'credit']['amount']
        if not credits.empty:
            # Average credit * 12 months = estimated yearly income
            avg_monthly_credit = credits.sum() / len(df)
            yearly_income = avg_monthly_credit * 12
        else:
            yearly_income = 0
        
        return {
            'ApplicantIncome': yearly_income,
            'CoapplicantIncome': 0,
            'LoanAmount': amount / 1000,
            'Loan_Amount_Term': 360,
            'Credit_History': 1 if (has_collateral or yearly_income > 100000) else 0
        }
        
    except Exception as e:
        print(f"Error reading user statement: {e}")
        # Return safe defaults on error
        return {
            'ApplicantIncome': 0,
            'CoapplicantIncome': 0,
            'LoanAmount': amount / 1000,
            'Loan_Amount_Term': 360,
            'Credit_History': 1 if has_collateral else 0
        }


def predict_loan_eligibility(user_id, amount, has_collateral, user_balance):
    """
    Main function called by loan view to predict loan eligibility.
    
    This function combines rule-based checks with ML prediction:
    1. Auto-approve if balance > 1 million (excellent standing)
    2. Use ML model for probability score
    3. Apply business rules for approval/pending/rejection
    
    Args:
        user_id: The user's ID
        amount: Loan amount requested
        has_collateral: Boolean indicating collateral
        user_balance: Current user balance from users.csv
        
    Returns:
        Tuple of (status, score, message) where:
        - status: 'approved', 'pending', or 'rejected'
        - score: ML probability score (0.0 to 1.0)
        - message: User-friendly message explaining the decision
    
    Note for Mohib (Intern):
        The rule-based auto-approval for high balance users is a simple
        business rule that doesn't need ML. This is faster and more
        explainable to the teacher.
    """
    # Rule 1: If balance > 1 million → auto eligible (excellent customer)
    if float(user_balance) >= 1000000:
        return (
            'approved',
            1.0,
            'Congratulations! Based on your excellent account standing, your loan is approved.'
        )
    
    # Get features for ML model
    features = compute_user_features(user_id, amount, has_collateral)
    feature_df = pd.DataFrame([features])
    
    # Check if model is loaded
    if loan_model is None:
        # Fallback to rule-based decision if model not available
        if has_collateral:
            return ('pending', 0.5, 'Your application is under review. Since you have collateral, an admin will review your request.')
        else:
            return ('rejected', 0.3, 'Based on your account history, we cannot approve this loan at this time.')
    
    # Get probability score from ML model
    # predict_proba returns [prob_rejected, prob_approved]
    prob = loan_model.predict_proba(feature_df)[0][1]  # probability of approval
    
    # Apply decision rules
    if prob >= 0.6:
        return ('approved', prob, 'Congratulations! Your loan application has been approved.')
    elif has_collateral:
        return ('pending', prob, 'Your application is under review. Since you have collateral, an admin will review your request.')
    else:
        return ('rejected', prob, 'Based on your account history, we cannot approve this loan at this time.')


# ══════════════════════════════════════════════════════════════════════
#  FRAUD DETECTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def detect_fraud(user_id, amount, hour_of_day):
    """
    Detect potential fraud for a transaction.
    
    This function combines rule-based checks with ML anomaly detection:
    1. Rule-based: Large transactions, unusual hours
    2. ML-based: Isolation Forest anomaly detection
    
    Args:
        user_id: The user's ID
        amount: Transaction amount
        hour_of_day: Hour of transaction (0-23)
        
    Returns:
        Tuple of (is_fraud, reason) where:
        - is_fraud: Boolean indicating if fraud is detected
        - reason: Explanation string (empty if no fraud)
    
    Note for Mohib (Intern):
        We use rule-based checks FIRST because they are:
        1. Easy to explain to the teacher
        2. Fast and deterministic
        3. Provide clear reasons for flagging
        
        ML is used as a secondary check for patterns rules might miss.
    """
    reasons = []
    
    # ── Rule-based checks (simple, explainable) ───────────────────────
    
    # Rule 1: Very large transaction (> 500,000 PKR)
    if amount > 500000:
        reasons.append("Unusually large transaction amount")
    
    # Rule 2: Late night transaction (12AM-5AM) with large amount (> 100k)
    if 0 <= hour_of_day <= 5 and amount > 100000:
        reasons.append("Large transaction during unusual hours (12AM-5AM)")
    
    # ── ML-based check (anomaly detection) ─────────────────────────────
    if fraud_artifacts is not None:
        fraud_model = fraud_artifacts['model']
        fraud_scaler = fraud_artifacts['scaler']
        
        # Prepare features in same format as training
        # Note: hour * 3600 converts to seconds (matching training format)
        features = np.array([[amount, hour_of_day * 3600]])
        features_scaled = fraud_scaler.transform(features)
        
        # Isolation Forest: -1 = anomaly (fraud), 1 = normal
        prediction = fraud_model.predict(features_scaled)[0]
        
        if prediction == -1:
            reasons.append("ML model detected anomalous pattern")
    
    # Determine if fraud based on any triggered rules
    is_fraud = len(reasons) > 0
    reason = '; '.join(reasons) if reasons else ''
    
    return is_fraud, reason


def flag_transaction(user_id, transaction_id, reason, severity='medium'):
    """
    Save a fraud flag to fraud_flags.csv for admin review.
    
    Args:
        user_id: The user's ID
        transaction_id: The transaction ID that was flagged
        reason: Explanation of why it was flagged
        severity: 'low', 'medium', or 'high'
    
    Note for Mohib (Intern):
        This function appends to the CSV file. We read the entire file,
        add the new row, and write it back. This is the pattern we use
        for all CSV operations in this project.
    """
    print(f"Saving fraud flag → user:{user_id} txn:{transaction_id} reason:{reason}")
    
    try:
        df = pd.read_csv(FRAUD_CSV)
    except Exception:
        # Create empty DataFrame if file doesn't exist
        df = pd.DataFrame(columns=[
            'flag_id', 'user_id', 'transaction_id',
            'reason', 'flagged_at', 'resolved', 'severity'
        ])
    
    # Generate new flag_id (max + 1, or 1 if empty)
    new_id = int(df['flag_id'].max()) + 1 if not df.empty else 1
    
    # Create new flag record
    new_flag = {
        'flag_id': new_id,
        'user_id': user_id,
        'transaction_id': transaction_id,
        'reason': reason,
        'flagged_at': str(datetime.datetime.now()),
        'resolved': False,
        'severity': severity,
    }
    
    # Append and save
    df = pd.concat([df, pd.DataFrame([new_flag])], ignore_index=True)
    df.to_csv(FRAUD_CSV, index=False)
    print(f"fraud_flags.csv updated — total flags: {len(df)}")
