"""
Loan Views - Loan Application and Status API Endpoints

This module handles:
- Loan application with ML-based eligibility prediction
- Loan status checking for users

Author: Sharjeel (as per sharjeel.md assignment)
"""

import os
import pandas as pd
import datetime

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from smartbank.authentication import CSVTokenAuthentication
from .ml_model import predict_loan_eligibility

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
LOANS_CSV = os.path.join(DATA_DIR, 'loans.csv')


# ── Helper Functions ───────────────────────────────────────────────────

def get_all_loans():
    """
    Read all loans from loans.csv.
    
    Returns:
        pandas DataFrame with all loan records
    """
    try:
        return pd.read_csv(LOANS_CSV)
    except Exception:
        # Return empty DataFrame with expected columns if file doesn't exist
        return pd.DataFrame(columns=[
            'loan_id', 'user_id', 'amount', 'purpose', 'duration_months',
            'has_collateral', 'asset_description', 'status', 'applied_at',
            'approved_at', 'ml_score'
        ])


def save_loan(loan_dict):
    """
    Save a new loan record to loans.csv.
    
    Args:
        loan_dict: Dictionary containing loan data
    
    Note for Mohib (Intern):
        We read the entire CSV, append the new row, and write back.
        This is the standard pattern for CSV operations in this project.
    """
    df = get_all_loans()
    new_row = pd.DataFrame([loan_dict])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(LOANS_CSV, index=False)


def has_active_loan(user_id):
    """
    Check if user already has an active (non-rejected) loan.
    
    Args:
        user_id: The user's ID
        
    Returns:
        True if user has active loan, False otherwise
    
    Note for Mohib (Intern):
        A loan is considered "active" if its status is not 'rejected'.
        This includes 'approved', 'pending', and any other non-rejected status.
    """
    df = get_all_loans()
    if df.empty:
        return False
    
    # Filter by user_id and status != 'rejected'
    user_loans = df[
        (df['user_id'] == int(user_id)) & 
        (df['status'].astype(str).str.lower() != 'rejected')
    ]
    return not user_loans.empty


# ══════════════════════════════════════════════════════════════════════
#  POST /api/loans/apply/
# ══════════════════════════════════════════════════════════════════════

class LoanApplyView(APIView):
    """
    Handle loan applications with ML-based eligibility prediction.
    
    Flow:
    1. Authenticate user via token
    2. Check if user already has active loan
    3. Get loan details from request
    4. Call ML prediction function
    5. Save loan record with status from ML
    6. Return status and message to user
    """
    authentication_classes = [CSVTokenAuthentication]
    
    def post(self, request):
        # ── 1. Get authenticated user ──────────────────────────────────
        # Check if authentication was successful
        if not hasattr(request, 'user') or not isinstance(request.user, dict):
            return Response(
                {'error': 'Authentication required. Please login first.'},
                status=status.HTTP_401_UNAUTHORIZED
            )
            
        user = request.user  # dict from CSV: { user_id, name, balance, ... }
        user_id = user['user_id']
        user_balance = user['balance']
        
        # ── 2. Check if user already has active loan ───────────────────
        if has_active_loan(user_id):
            return Response(
                {'error': 'You already have an active loan. Please complete it before applying for a new one.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # ── 3. Get loan details from request ───────────────────────────
        # Required fields
        amount = request.data.get('amount')
        purpose = request.data.get('purpose', 'personal')
        duration_months = request.data.get('duration_months', 12)
        has_collateral = request.data.get('has_collateral', False)
        
        # Optional field
        asset_description = request.data.get('asset_description', '')
        
        # Validate required fields
        if amount is None:
            return Response(
                {'error': 'Loan amount is required.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            amount = float(amount)
            duration_months = int(duration_months)
            has_collateral = bool(has_collateral)
        except (ValueError, TypeError):
            return Response(
                {'error': 'Invalid data types. Amount and duration must be numbers.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if amount <= 0:
            return Response(
                {'error': 'Loan amount must be positive.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # ── 4. Call ML prediction function ─────────────────────────────
        loan_status, ml_score, message = predict_loan_eligibility(
            user_id=user_id,
            amount=amount,
            has_collateral=has_collateral,
            user_balance=user_balance
        )
        
        # ── 5. Save loan record to loans.csv ───────────────────────────
        df = get_all_loans()
        new_loan_id = int(df['loan_id'].max()) + 1 if not df.empty else 1
        
        loan_record = {
            'loan_id': new_loan_id,
            'user_id': user_id,
            'amount': amount,
            'purpose': purpose,
            'duration_months': duration_months,
            'has_collateral': has_collateral,
            'asset_description': asset_description,
            'status': loan_status,
            'applied_at': str(datetime.datetime.now()),
            'approved_at': '',  # Will be filled when admin approves (if pending)
            'ml_score': round(ml_score, 4)
        }
        
        save_loan(loan_record)
        
        # ── 6. Return response to user ─────────────────────────────────
        return Response(
            {
                'message': message,
                'data': {
                    'loan_id': new_loan_id,
                    'status': loan_status,
                    'amount': amount,
                    'ml_score': round(ml_score, 4)
                }
            },
            status=status.HTTP_200_OK
        )


# ══════════════════════════════════════════════════════════════════════
#  GET /api/loans/status/
# ══════════════════════════════════════════════════════════════════════

class LoanStatusView(APIView):
    """
    Get user's current loan status.
    
    Returns the latest non-rejected loan for the authenticated user.
    If no active loan exists, returns null.
    """
    authentication_classes = [CSVTokenAuthentication]
    
    def get(self, request):
        # ── 1. Get authenticated user ──────────────────────────────────
        # Check if authentication was successful
        if not hasattr(request, 'user') or not isinstance(request.user, dict):
            return Response(
                {'error': 'Authentication required. Please login first.'},
                status=status.HTTP_401_UNAUTHORIZED
            )
            
        user = request.user
        user_id = user['user_id']
        
        # ── 2. Read loans.csv ──────────────────────────────────────────
        df = get_all_loans()
        
        if df.empty:
            return Response({'loan': None}, status=status.HTTP_200_OK)
        
        # ── 3. Filter by user_id and status != 'rejected' ──────────────
        # Convert user_id to match CSV type (could be int or str)
        user_loans = df[
            (df['user_id'].astype(str) == str(user_id)) & 
            (df['status'].astype(str).str.lower() != 'rejected')
        ]
        
        if user_loans.empty:
            return Response({'loan': None}, status=status.HTTP_200_OK)
        
        # ── 4. Return latest loan (most recent applied_at) ─────────────
        user_loans = user_loans.sort_values('applied_at', ascending=False)
        latest_loan = user_loans.iloc[0].to_dict()
        
        # Clean up the response (remove unnecessary fields)
        loan_response = {
            'loan_id': int(latest_loan['loan_id']),
            'amount': float(latest_loan['amount']),
            'purpose': latest_loan['purpose'],
            'duration_months': int(latest_loan['duration_months']),
            'has_collateral': bool(latest_loan['has_collateral']),
            'status': latest_loan['status'],
            'applied_at': latest_loan['applied_at'],
            'ml_score': float(latest_loan['ml_score']) if latest_loan['ml_score'] else None
        }
        
        # Add optional fields if they exist
        if latest_loan.get('asset_description'):
            loan_response['asset_description'] = latest_loan['asset_description']
        if latest_loan.get('approved_at'):
            loan_response['approved_at'] = latest_loan['approved_at']
        
        return Response({'loan': loan_response}, status=status.HTTP_200_OK)
