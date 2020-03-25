from django.urls import path
from .views import (LMRTView)

urlpatterns = [
    path('manual/', LMRTView.as_view(), name='lmrt-manual'),
    path('automatic/', LMRTView.as_view(), name='lmrt-auto'),
    path('groundtruth/', LMRTView.as_view(), name='lmrt-gt'),
]
