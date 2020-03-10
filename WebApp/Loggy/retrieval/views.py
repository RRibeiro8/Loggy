from django.shortcuts import render
from django.views import View
from django.http import JsonResponse
#from .models import TopicObject
#from .forms import ObjectsForm
from fileupload.models import ImageModel, LocationModel
from visualrecognition.models import (ActivityModel, AttributesModel, CategoryModel, ConceptModel)
from django.core import serializers



class LMRTView(View):

	template_name = "retrieval/lmrt.html"

	def post(self, request, *args, **kwargs):

		if request.is_ajax():

			objects = request.POST.getlist('obj_tags[]')
			locations = request.POST.getlist('loc_tags[]')
			activities = request.POST.getlist('act_tags[]')
			others = request.POST.getlist('other_tags[]')

			#images_set = ImageModel.objects.all()

			image_list = {}

			for obj in objects:
				queryset = ConceptModel.objects.filter(tag=obj)
				for q in queryset:
					url = q.image.file.url
					name = q.image.file.name

					image_list[name] = url

			#serializers.serialize('json', image_list)
			#print(image_list)

			return JsonResponse({"success": True, "queryset": image_list}, status=200)
		else:
			return JsonResponse({"success": False}, status=400)

	def get(self, request, *args, **kwargs):

		context = {}
		return render(self.request, self.template_name, context)


