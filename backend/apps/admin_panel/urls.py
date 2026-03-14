from django.urls import path

urlpatterns = [
    path('users/',        AdminUsersView.as_view()),
    path('transactions/', AdminAllTransactionsView.as_view()),
]
