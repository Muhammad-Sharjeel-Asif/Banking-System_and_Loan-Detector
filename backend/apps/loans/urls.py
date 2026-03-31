from django.urls import path
from .views import LoanApplyView, LoanStatusView

urlpatterns = [
    path('apply/', LoanApplyView.as_view(), name='loan-apply'),
    path('status/', LoanStatusView.as_view(), name='loan-status'),
]
