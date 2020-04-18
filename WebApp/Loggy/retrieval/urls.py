from django.urls import path
from .views import (LMRTView, GTView, LMRTConnectionsView)

urlpatterns = [
    path('manual/', LMRTView.as_view(), name='lmrt-manual'),
    path('manual/connections/', LMRTConnectionsView.as_view(), name='lmrt-manual-connected'),
    path('automatic/', LMRTView.as_view(), name='lmrt-auto'),
    path('groundtruth/', GTView.as_view(), name='lmrt-gt'),
]
