import os
import glob
import pandas as pd

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