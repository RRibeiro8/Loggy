from django.shortcuts import render
from django.views import View
from django.http import JsonResponse
#from .models import TopicObject
#from .forms import ObjectsForm
from fileupload.models import ImageModel, LocationModel
from visualrecognition.models import (ActivityModel, AttributesModel, CategoryModel, ConceptModel)
from django.core import serializers

from .models import TopicModel

from collections import Counter
from .sentence_analyzer.nlp_analyzer import similarity

from tqdm import tqdm

import os
from django.conf import settings


class LMRTView(View):

	template_name = "retrieval/lmrt.html"

	def post(self, request, *args, **kwargs):

		if request.is_ajax():

			objects = request.POST.getlist('obj_tags[]')
			locations = request.POST.getlist('loc_tags[]')
			activities = request.POST.getlist('act_tags[]')
			others = request.POST.getlist('other_tags[]')

			images_set = ImageModel.objects.all()
			image_list = {}
			queryset = {}
		
			for img in tqdm(images_set):
				
				concepts_sim_score = self.compute_score(img.conceptmodel_set.all(), objects)
				location_sim_score = self.compute_score(img.locationmodel_set.all(), locations)
				activities_sim_score = self.compute_score(img.activitymodel_set.all(), activities)

				img_sim_score = (concepts_sim_score + activities_sim_score + location_sim_score) / 3
				#print(concepts_sim_score, activities_sim_score, location_sim_score, img_sim_score)
				
				if img_sim_score > 0.6:

					#queryset[img.slug] = img_sim_score
					url = img.file.url
					name = img.file.name

					image_list[name] = url

			#serializers.serialize('json', image_list)
			#print(image_list)

			return JsonResponse({"success": True, "queryset": image_list}, status=200)
		else:
			return JsonResponse({"success": False}, status=400)

	def get(self, request, *args, **kwargs):

		context = {}
		return render(self.request, self.template_name, context)


	def compute_score(self, imgs, words):

		count_concepts = Counter([c.tag for c in imgs])

		sim_score = 0
		
		for obj in words:
			obj_sim_score = 0
			for con in count_concepts:
			
				score = similarity(obj, con)

				if score > obj_sim_score:
					obj_sim_score = score

			sim_score += obj_sim_score

		if len(words) > 0:
			sim_score = sim_score / len(words)

		return sim_score

class GTView(View):

	template_name = "retrieval/gt.html"
	model = TopicModel

	def get(self, request, *args, **kwargs):

		topicset = self.model.objects.all()
		context = { 'topics': topicset }
		return render(self.request, self.template_name, context)

	def post(self, request, *args, **kwargs):

		if request.is_ajax():

			topic_title = request.POST.getlist('data')[0]

			obj = self.model.objects.filter(title=topic_title)[0]

			image_list = {}

			with open(os.path.join(settings.MEDIA_ROOT, 'ImageCLEF2020_dev_gt.txt'), 'r') as f:
				lines = f.readlines()

				for l in lines:
					tmp = l.split(', ')

					if tmp[0] == obj.topic_id:

						img = ImageModel.objects.filter(slug=tmp[1])[0]
						url = img.file.url
						name = img.file.name

						image_list[name] = url


			context = { "success": True,
						"title": obj.title,
						"narrative": obj.narrative,
						"description": obj.description,
						"queryset": image_list
						}

			return JsonResponse(context, status=200)
		else:
			return JsonResponse({"success": False}, status=400)



