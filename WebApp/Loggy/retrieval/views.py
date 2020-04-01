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
from .sentence_analyzer.nlp_analyzer import similarity, word2lemma

from tqdm import tqdm
import numpy as np

import os
from django.conf import settings

import time


class LMRTView(View):

	template_name = "retrieval/lmrt.html"

	def post(self, request, *args, **kwargs):

		if request.is_ajax():

			objects = request.POST.getlist('obj_tags[]')
			locations = request.POST.getlist('loc_tags[]')
			activities = request.POST.getlist('act_tags[]')
			timedates = request.POST.getlist('other_tags[]')

			images_set = ImageModel.objects.all()
			image_list = {}
			queryset = {}

			#print(new_objs)
			#print(locations)
			#print(new_act)
			#print(timedates)
			### 
			
			tic = time.clock()

			for img in tqdm(images_set):
				
				### processing time -- 0.004 s
				count_concepts = self.word_counter(img.conceptmodel_set.all())
				count_location = self.word_counter(img.locationmodel_set.all())
				count_activities = self.word_counter(img.activitymodel_set.all())
				count_attributes = self.word_counter(img.attributesmodel_set.all())
				count_categories = self.word_counter(img.categorymodel_set.all())
				###
				print(img.file.name)

				### processing time -- 0.01 s
				concepts_score = self.retrieve_scores(count_concepts, img.conceptmodel_set)
				categories_score = self.retrieve_scores(count_categories, img.categorymodel_set)
				#####

				tic = time.clock()

				d = {**concepts_score}
				final_objs_score = self.compute_score(objects, d)

				d = {**categories_score, **count_location}		
				final_locations_score = self.compute_score(locations, d)

				img_conf = (final_locations_score + final_objs_score) / 2 
				#print("Final: ", final_objs_score, final_locations_score, img_conf)
				if img_conf > 0:

					url = img.file.url
					name = img.file.name

					score = img_conf*100
					image_list[name] = [ {'url': url, 'conf': score} ]

				toc = time.clock()
				#print("Processing time: ", (toc - tic))


			return JsonResponse({"success": True, "queryset": image_list}, status=200)
		else:
			return JsonResponse({"success": False}, status=400)

	def get(self, request, *args, **kwargs):

		context = {}
		return render(self.request, self.template_name, context)

	def retrieve_scores(self, counts, img_at):
		toreturn = {}
		for con in counts:
			#### If we want to take into account the number of objects 
			#if counts[con] == 1 or > 1:
			querytag = img_at.filter(tag=con)
			for c in querytag:
				lemma = word2lemma(c.tag)
				if len(lemma) > 1:
					n_gram = ""
					for l in lemma:
						if n_gram == "":
							n_gram = l
						else:
							n_gram = n_gram + " " + l
					
					if n_gram not in toreturn:
						toreturn[n_gram] = c.score
					else:
						if c.score > toreturn[n_gram]:
							toreturn[n_gram] = c.score
				else:

					for l in lemma:
						if l not in toreturn:
							toreturn[l] = c.score
						else:
							if c.score > toreturn[l]:
								toreturn[l] = c.score

		return toreturn

	def word_counter(self, img_words):

		lista = []
		for c in img_words:
			if c.tag != "NULL":
				lista.append(c.tag)
		return Counter(lista)

	def compute_score(self, objects, d):
		final_objs_score = 0
		for word in objects:
			w_lemma = word2lemma(word)
			w_score = 0
			i = 0
			for con in d:
				##if we wanto to filter more, we can treshold the similarity
				sim_score = similarity(w_lemma[0], con)
				if sim_score < 0.5:
					sim_score = 0

				con_score = d[con]*sim_score
				#print(w_lemma[0], con, con_score)

				if con_score >= 0.4:
					w_score = (w_score + con_score)
					i += 1

			if i > 0:
				w_score = w_score / i

			#print(w_lemma[0], w_score)

			final_objs_score = final_objs_score + w_score
		
		if len(objects) > 0:
			final_objs_score = final_objs_score / len(objects)
		
		return final_objs_score

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



