from fileupload.models import (ImageModel, LocationModel,ConceptModel,ConceptScoreModel,LocationModelAdmin, 
								ImageModelAdmin,ConceptModelAdmin, CategoryModel, CategoryScoreModel, 
								CategoryModelAdmin, LocationInfoModel, ActivityModel, ActivityModelAdmin,
								ActivityInfoModel, AttributesModel, AttributesModelAdmin, AttributesInfoModel)
from django.contrib import admin

admin.site.register(ImageModel, ImageModelAdmin)
admin.site.register(LocationModel)#, LocationModelAdmin)
#admin.site.register(LocationInfoModel)
admin.site.register(ConceptModel)#, ConceptModelAdmin)
#admin.site.register(ConceptScoreModel)
admin.site.register(CategoryModel)#, CategoryModelAdmin)
#admin.site.register(CategoryScoreModel)
admin.site.register(ActivityModel)#, ActivityModelAdmin)
#admin.site.register(ActivityInfoModel)
admin.site.register(AttributesModel)#, AttributesModelAdmin)
#admin.site.register(AttributesInfoModel)
