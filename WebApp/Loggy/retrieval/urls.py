from django.urls import path
from .views import (LMRTView)

urlpatterns = [
    path('', LMRTView.as_view(), name='lmrt'),
]
