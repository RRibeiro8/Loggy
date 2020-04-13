from django.contrib import admin
from .models import TopicModel, SimilarityModel

admin.site.register(TopicModel)
admin.site.register(SimilarityModel)
