from django.urls import path
from .views import (GalleryView)

urlpatterns = [
 path('', GalleryView.as_view(), name='gallery'),
]
