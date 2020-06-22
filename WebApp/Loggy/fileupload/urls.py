from django.urls import path
from fileupload.views import (ImageCreateView, ImageDeleteView, ImageListView, UpdateConceptsView)

urlpatterns = [

    path('new/', ImageCreateView.as_view(), name='upload-new'),
    path('delete/<int:pk>', ImageDeleteView.as_view(), name='upload-delete'),
    path('view/', ImageListView.as_view(), name='upload-view'),
    path('update/concepts/', UpdateConceptsView.as_view(), name='upload-concepts'),
]
