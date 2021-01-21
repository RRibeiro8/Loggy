from django.urls import path
from fileupload.views import (ImageCreateView, ImageDeleteView, ImageListView, UpdateConceptsView, UpdateActivitiesView, UpdateLocationsView)

urlpatterns = [

    path('new/', ImageCreateView.as_view(), name='upload-new'),
    path('delete/<int:pk>', ImageDeleteView.as_view(), name='upload-delete'),
    path('view/', ImageListView.as_view(), name='upload-view'),
    path('update/concepts/', UpdateConceptsView.as_view(), name='upload-concepts'),
    path('update/activities/', UpdateActivitiesView.as_view(), name='upload-activities'),
    path('update/locations/', UpdateLocationsView.as_view(), name='upload-locations'),
]
