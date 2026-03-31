import os
import glob
import pandas as pd
import datetime

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from smartbank.authentication import CSVTokenAuthentication

# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), '../../data')
USERS_CSV = os.path.join(DATA_DIR, 'users.csv')
STMTS_DIR = os.path.join(DATA_DIR, 'statements')


# ── shared admin check helper ──────────────────────────────────────────
def is_admin(request):
    return request.user.get('role') == 'admin'


# ══════════════════════════════════════════════════════════════════════
#  GET /api/admin/users/
# ══════════════════════════════════════════════════════════════════════

class AdminUsersView(APIView):
    authentication_classes = [CSVTokenAuthentication]

    def get(self, request):

        # ── 1. Admin check ─────────────────────────────────────────
        if not is_admin(request):
            return Response(
                {'error': 'Access denied. Admins only.'},
                status=status.HTTP_403_FORBIDDEN
            )

        # ── 2. Read users.csv ──────────────────────────────────────
        df = pd.read_csv(USERS_CSV)

        # ── 3. Drop password_hash column ──────────────────────────
        df = df.drop(columns=['password_hash'])

        # ── 4. Return as list of dicts ─────────────────────────────
        users = df.to_dict(orient='records')

        return Response(users, status=status.HTTP_200_OK)


# ══════════════════════════════════════════════════════════════════════
#  GET /api/admin/transactions/
# ══════════════════════════════════════════════════════════════════════

class AdminAllTransactionsView(APIView):
    authentication_classes = [CSVTokenAuthentication]

    def get(self, request):

        # ── 1. Admin check ─────────────────────────────────────────
        if not is_admin(request):
            return Response(
                {'error': 'Access denied. Admins only.'},
                status=status.HTTP_403_FORBIDDEN
            )

        # ── 2. Find all user_*.csv files ───────────────────────────
        pattern = os.path.join(STMTS_DIR, 'user_*.csv')
        all_files = glob.glob(pattern)

        if not all_files:
            return Response([], status=status.HTTP_200_OK)

        # ── 3. Loop through each file, add user_id column, collect ─
        frames = []
        for filepath in all_files:
            try:
                df = pd.read_csv(filepath)
                if df.empty:
                    continue

                # Extract user_id from filename  e.g. user_3.csv → 3
                filename = os.path.basename(filepath)          # user_3.csv
                user_id  = filename.replace('user_', '').replace('.csv', '')
                df['user_id'] = user_id

                frames.append(df)

            except Exception:
                continue   # skip corrupted files

        if not frames:
            return Response([], status=status.HTTP_200_OK)

        # ── 4. Concatenate all into one DataFrame ──────────────────
        combined = pd.concat(frames, ignore_index=True)

        # ── 5. Attach user name so admin can see who made transfer ─
        users_df = pd.read_csv(USERS_CSV)[['user_id', 'name']]
        users_df['user_id'] = users_df['user_id'].astype(str)
        combined['user_id'] = combined['user_id'].astype(str)
        combined = combined.merge(users_df, on='user_id', how='left')

        # ── 6. Sort by date descending (most recent first) ─────────
        combined = combined.sort_values('date', ascending=False)

        # ── 7. Replace NaN with empty string (safe for JSON) ───────
        combined = combined.fillna('')

        return Response(combined.to_dict(orient='records'), status=status.HTTP_200_OK)

# ══════════════════════════════════════════════════════════════════════
#  GET /api/admin/fraud-alerts/
# ══════════════════════════════════════════════════════════════════════

FRAUD_CSV = os.path.join(DATA_DIR, 'fraud_flags.csv')

class AdminFraudAlertsView(APIView):
    authentication_classes = [CSVTokenAuthentication]

    def get(self, request):

        # ── 1. Admin check ─────────────────────────────────────────
        if not is_admin(request):
            return Response(
                {'error': 'Access denied. Admins only.'},
                status=status.HTTP_403_FORBIDDEN
            )

        # ── 2. Read fraud_flags.csv ────────────────────────────────
        try:
            flags_df = pd.read_csv(FRAUD_CSV)
        except Exception:
            return Response([], status=status.HTTP_200_OK)

        if flags_df.empty:
            return Response([], status=status.HTTP_200_OK)

        # ── 3. Filter only unresolved flags ────────────────────────
        # resolved column stores True/False as string in CSV
        # so we handle both cases
        flags_df['resolved'] = flags_df['resolved'].astype(str).str.strip().str.lower()
        unresolved = flags_df[flags_df['resolved'] == 'false']

        if unresolved.empty:
            return Response([], status=status.HTTP_200_OK)

        # ── 4. Join with users.csv to get user name ────────────────
        users_df = pd.read_csv(USERS_CSV)[['user_id', 'name', 'email']]
        users_df['user_id'] = users_df['user_id'].astype(str)
        unresolved = unresolved.copy()
        unresolved['user_id'] = unresolved['user_id'].astype(str)

        merged = unresolved.merge(users_df, on='user_id', how='left')

        # ── 5. Fill any missing values (safe for JSON) ─────────────
        merged = merged.fillna('')

        # ── 6. Sort by flagged_at descending (newest first) ────────
        merged = merged.sort_values('flagged_at', ascending=False)

        return Response(merged.to_dict(orient='records'), status=status.HTTP_200_OK)


# ══════════════════════════════════════════════════════════════════════
#  PUT /api/admin/loans/{id}/approve/
# ══════════════════════════════════════════════════════════════════════

LOANS_CSV = os.path.join(DATA_DIR, 'loans.csv')

def get_all_loans():
    """Read all loans from loans.csv."""
    try:
        return pd.read_csv(LOANS_CSV)
    except Exception:
        return pd.DataFrame(columns=[
            'loan_id', 'user_id', 'amount', 'purpose', 'duration_months',
            'has_collateral', 'asset_description', 'status', 'applied_at',
            'approved_at', 'ml_score'
        ])


def save_loans(df):
    """Save loans DataFrame to loans.csv."""
    df.to_csv(LOANS_CSV, index=False)


class AdminLoanApprovalView(APIView):
    """
    Admin endpoint to approve or reject loan applications.
    
    Only admins can access this endpoint.
    Updates the loan status and sets the approval timestamp.
    
    Note for Mohib (Intern):
        This view uses PUT method because we're updating an existing resource.
        The loan ID is passed as a URL parameter (path kwarg).
    """
    authentication_classes = [CSVTokenAuthentication]
    
    def put(self, request, loan_id):
        # ── 1. Admin check ─────────────────────────────────────────
        if not is_admin(request):
            return Response(
                {'error': 'Access denied. Admins only.'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        # ── 2. Get action from request (approve or reject) ─────────
        action = request.data.get('action')
        
        if action not in ['approve', 'reject']:
            return Response(
                {'error': "Invalid action. Must be 'approve' or 'reject'."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # ── 3. Read loans.csv and find the loan ────────────────────
        df = get_all_loans()
        
        if df.empty:
            return Response(
                {'error': 'Loan not found.'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Find loan by ID (handle both int and str types)
        loan_idx = df.index[df['loan_id'].astype(str) == str(loan_id)]
        
        if len(loan_idx) == 0:
            return Response(
                {'error': 'Loan not found.'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        loan_idx = loan_idx[0]
        
        # ── 4. Update loan status ──────────────────────────────────
        new_status = 'approved' if action == 'approve' else 'rejected'
        df.loc[loan_idx, 'status'] = new_status
        
        # Set approved_at timestamp only if approving
        if action == 'approve':
            df.loc[loan_idx, 'approved_at'] = str(datetime.datetime.now())
        
        # ── 5. Save back to CSV ────────────────────────────────────
        save_loans(df)
        
        # ── 6. Return updated loan info ────────────────────────────
        updated_loan = df.loc[loan_idx].to_dict()
        
        return Response(
            {
                'message': f'Loan {loan_id} has been {new_status}.',
                'data': {
                    'loan_id': int(updated_loan['loan_id']),
                    'user_id': int(updated_loan['user_id']),
                    'amount': float(updated_loan['amount']),
                    'status': updated_loan['status'],
                    'approved_at': updated_loan.get('approved_at', '')
                }
            },
            status=status.HTTP_200_OK
        )