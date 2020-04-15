from django.shortcuts import render
from django.views import View
from django.http import JsonResponse

from fileupload.models import (ImageModel, LocationModel, ConceptModel, ConceptScoreModel, 
								CategoryModel, CategoryScoreModel, ActivityInfoModel, ActivityModel, 
								AttributesModel, AttributesInfoModel)
from django.core import serializers

from .models import TopicModel, SimilarityModel

from .auxiliary import create_dict, compute_score, best_clusters, evaluation

from tqdm import tqdm
import numpy as np

import os
from django.conf import settings
from django.db.models import Q

import time
import collections
import itertools 
import datetime


class LMRTView(View):

	template_name = "retrieval/lmrt.html"

	def post(self, request, *args, **kwargs):

		if request.is_ajax():

			objects = request.POST.getlist('obj_tags[]')
			locations = request.POST.getlist('loc_tags[]')
			activities = request.POST.getlist('act_tags[]')
			negatives = request.POST.getlist('neg_tags[]')
			topic_id = request.POST.getlist('topic_id')[0]
			daterange = request.POST.getlist('daterange')
			years = request.POST.getlist('years[]')
			daysweek = request.POST.getlist('daysweek[]')

			image_list = {}
			queryset = {}
			evaluation_data = {}

			images_set =  None

			img_clusters = []

			if( objects or locations or activities or negatives):

				if( daterange or years or daysweek ):

					filtro = Q()

					if daterange:
						dates = daterange[0].split(' - ')
						start_date = datetime.datetime.strptime(dates[0], '%d/%m/%Y').strftime('%Y-%m-%d')
						end_date =  datetime.datetime.strptime(dates[0], '%d/%m/%Y')
						new_end = end_date + datetime.timedelta(days=1)

						filtro = filtro | Q(date_time__range=[start_date, new_end.strftime('%Y-%m-%d')])
					
					if years:
						for y in years:
							filtro = filtro | Q(date_time__year=y)

					if daysweek:
						for day in daysweek:
							filtro = filtro | Q(date_time__week_day=int(day))
						#print("Days of Week: ", daysweek)

					images_set = ImageModel.objects.filter(filtro)

				else:
					images_set = ImageModel.objects.all()

				evaluation_list = {}

				for img in tqdm(images_set):

					scores = []

					if negatives:

						img_concepts = img.concepts.all()
						d_concepts = create_dict(img_concepts, img, ConceptScoreModel.objects)
						#count_concepts = self.word_counter(img.conceptmodel_set.all())
						#concepts_score = self.retrieve_scores(count_concepts, img.conceptmodel_set)
						
						img_location = img.location.all()
						img_categories = img.categories.all()
						d_categoties = create_dict(img_categories, img, CategoryScoreModel.objects)
						d_locations = create_dict(img_location, img)
						#count_location = self.word_counter(img.locationmodel_set.all())
						#count_categories = self.word_counter(img.categorymodel_set.all())
						#categories_score = self.retrieve_scores(count_categories, img.categorymodel_set)

						img_activity = img.activities.all()
						img_attributes = img.attributes.all()
						d_activities = create_dict(img_activity, img)
						d_attributes = create_dict(img_attributes, img)

						#count_activities = self.word_counter(img.activitymodel_set.all())
						#count_attributes = self.word_counter(img.attributesmodel_set.all())


						d = {**d_concepts, **d_categoties, **d_activities, **d_attributes}
						neg_score = compute_score(negatives, d)

						if neg_score < 0.6:

							if objects:
								d = {**d_concepts}
								scores.append(compute_score(objects, d))

							if locations:
								d = {**d_categoties, **d_locations}		
								scores.append(compute_score(locations, d))
							
							if activities:
								#print("activities: ", activities)
								#att_verbs, att_nouns = self.attributes_filter(count_attributes)
								d = {**d_activities, **d_attributes}
								scores.append(compute_score(activities, d))		

					else:

						if objects:
							#print("Objects: ", objects)
							img_concepts = img.concepts.all()
							#img_categories = img.categories.all()
							d = {**create_dict(img_concepts, img, ConceptScoreModel.objects)}#,
							#**self.create_dict(img_categories, img, CategoryScoreModel.objects)}
							scores.append(compute_score(objects, d))

						if locations:
							#print("locations: ", locations)
							img_location = img.location.all()
							img_categories = img.categories.all()
							d = {**create_dict(img_location, img), 
								**create_dict(img_categories, img, CategoryScoreModel.objects)}
							
							scores.append(compute_score(locations, d))
						
						if activities:
							#print("activities: ", activities)
							img_activity = img.activities.all()
							img_attributes = img.attributes.all()
							#count_attributes = self.word_counter(img.attributesmodel_set.all())
							#att_verbs, att_nouns = self.attributes_filter(count_attributes)
							d = {**create_dict(img_activity, img), **create_dict(img_attributes, img)}
							scores.append(compute_score(activities, d))

					img_conf = 0
					for s in scores:
						img_conf = img_conf + s
					
					if (len(scores) > 0):
						img_conf = img_conf / len(scores)
					
					if img_conf > 0.2:

						img_clusters.append((img, img_conf))
						evaluation_list[img.slug] = { "image": img, "confidence": img_conf }

					#toc = time.clock()
					#print("Processing time: ", (toc - tic))

				evaluation_data = {}
				
				clusters = best_clusters(img_clusters)

				image_list = {}

				for c in clusters:
					name = clusters[c]["image"].slug
					url =  clusters[c]["image"].file.url
					score = clusters[c]["confidence"]*100
					image_list[name] = [{'url': url, 'conf': score}]

				#print(image_list)

				if evaluation_list:
					img_list_sorted = sorted(evaluation_list.items(), key = lambda item: item[1]['confidence'], reverse=True)

					for item in img_list_sorted:

						if item[0] not in image_list:
							name = item[1]["image"].slug
							url =  item[1]["image"].file.url
							score = item[1]["confidence"]*100
							image_list[name] = [{'url': url, 'conf': score}]
							#print(item[0], image_list[item[0]])

					#print(img_list_sorted)

					recall, precision, f1_score = evaluation(dict(itertools.islice(image_list.items(), 5)), topic_id)
					evaluation_data["Top5"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]
					recall, precision, f1_score = evaluation(dict(itertools.islice(image_list.items(), 10)), topic_id)
					evaluation_data["Top10"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]
					recall, precision, f1_score = evaluation(dict(itertools.islice(image_list.items(), 20)), topic_id)
					evaluation_data["Top20"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]
					recall, precision, f1_score = evaluation(dict(itertools.islice(image_list.items(), 30)), topic_id)
					evaluation_data["Top30"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]
					recall, precision, f1_score = evaluation(dict(itertools.islice(image_list.items(), 40)), topic_id)
					evaluation_data["Top40"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]
					recall, precision, f1_score = evaluation(dict(itertools.islice(image_list.items(), 50)), topic_id)
					evaluation_data["Top50"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]


			return JsonResponse({"success": True, "queryset": image_list, "evaluation": evaluation_data}, status=200)
		else:
			return JsonResponse({"success": False}, status=400)

	def get(self, request, *args, **kwargs):

		topicset = TopicModel.objects.all()
		context = { 'topics': topicset }
		return render(self.request, self.template_name, context)

	
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

						img_objs = ImageModel.objects.filter(slug=tmp[1])
						if len(img_objs) > 0:
							img = img_objs[0]

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



