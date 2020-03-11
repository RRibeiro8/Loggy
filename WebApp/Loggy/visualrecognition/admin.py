from django.contrib import admin
from .models import (ActivityModel, AttributesModel, CategoryModel, ConceptModel)

admin.site.register(ActivityModel)
admin.site.register(AttributesModel)
admin.site.register(CategoryModel)
admin.site.register(ConceptModel)
