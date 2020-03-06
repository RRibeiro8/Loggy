from django.urls import path
from .views import (GalleryView, DateTimeView)

urlpatterns = [
 path('', GalleryView.as_view(), name='gallery'),
 path('timeline/', DateTimeView.as_view(), name='datetime'),
]
