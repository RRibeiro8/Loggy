from fileupload.models import ImageModel, LocationModel,ConceptModel,ConceptScoreModel, ImageModelAdmin,ConceptModelAdmin
from django.contrib import admin

admin.site.register(ImageModel, ImageModelAdmin)
admin.site.register(LocationModel)
admin.site.register(ConceptModel, ConceptModelAdmin)
admin.site.register(ConceptScoreModel)