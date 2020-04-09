from django.shortcuts import render
from django.views import View
from django.http import JsonResponse

from fileupload.models import ImageModel, LocationModel
from visualrecognition.models import (ActivityModel, AttributesModel, CategoryModel, ConceptModel)
from django.core import serializers

from .models import TopicModel

from collections import Counter
from .sentence_analyzer.nlp_analyzer import similarity, word2lemma, word2lemma_pos

from tqdm import tqdm
import numpy as np

import os
from django.conf import settings

import time
import collections


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

			evaluation_list = {}

			for img in tqdm(images_set):

				#print(img.date_time)
				
				### processing time -- 0.004 s
				count_concepts = self.word_counter(img.conceptmodel_set.all())
				count_location = self.word_counter(img.locationmodel_set.all())
				count_activities = self.word_counter(img.activitymodel_set.all())
				count_attributes = self.word_counter(img.attributesmodel_set.all())
				count_categories = self.word_counter(img.categorymodel_set.all())
				###
				#print(img.file.name)

				### processing time -- 0.01 s
				concepts_score = self.retrieve_scores(count_concepts, img.conceptmodel_set)
				categories_score = self.retrieve_scores(count_categories, img.categorymodel_set)
				#####

				att_verbs, att_nouns = self.attributes_filter(count_attributes)

				#tic = time.clock()

				d = {**concepts_score}
				final_objs_score = self.compute_score(objects, d)

				d = {**categories_score, **count_location}		
				final_locations_score = self.compute_score(locations, d)

				d = {**count_activities, **att_verbs}
				final_activities_score = self.compute_score(activities, d)

				img_conf = (final_locations_score + final_objs_score + final_activities_score) / 3
				#print("Final: ", final_objs_score, final_locations_score, img_conf)
				if img_conf > 0:

					url = img.file.url
					name = img.file.name

					evaluation_list[img.slug] = img_conf

					score = img_conf*100
					image_list[name] = [ {'url': url, 'conf': score} ]

				#toc = time.clock()
				#print("Processing time: ", (toc - tic))

			evaluation_data = {}
			img_list_sorted = sorted(evaluation_list.items(), key = lambda item: item[1], reverse=True)

			topic_id = "2"
			recall, precision, f1_score = self.evaluation(img_list_sorted[0:5], topic_id)
			evaluation_data["Top5"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]
			recall, precision, f1_score = self.evaluation(img_list_sorted[0:10], topic_id)
			evaluation_data["Top10"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]
			recall, precision, f1_score = self.evaluation(img_list_sorted[0:20], topic_id)
			evaluation_data["Top20"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]
			recall, precision, f1_score = self.evaluation(img_list_sorted[0:30], topic_id)
			evaluation_data["Top30"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]
			recall, precision, f1_score = self.evaluation(img_list_sorted[0:40], topic_id)
			evaluation_data["Top40"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]
			recall, precision, f1_score = self.evaluation(img_list_sorted[0:50], topic_id)
			evaluation_data["Top50"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]


			return JsonResponse({"success": True, "queryset": image_list, "evaluation": evaluation_data}, status=200)
		else:
			return JsonResponse({"success": False}, status=400)

	def get(self, request, *args, **kwargs):

		context = {}
		return render(self.request, self.template_name, context)

	def evaluation(self, img_list, topic_id):

		TP = 0
		total_clusters = []
		clusters = []
		X = len(img_list)

		with open(os.path.join(settings.MEDIA_ROOT, 'ImageCLEF2020_dev_gt.txt'), 'r') as f:
				lines = f.readlines()

				for l in lines:
					tmp = l.replace("\n", "")
					tmp = tmp.split(', ')
					
					if tmp[0] == topic_id:
						#print(tmp[1])
						if tmp[2] not in total_clusters:
							total_clusters.append(tmp[2])
						
						for img in img_list:
							#print(tmp[1], img[0])
							if tmp[1] == img[0]:
								if tmp[2] not in clusters:
									clusters.append(tmp[2])
								#print("positivo", img[0], tmp[2])
								TP = TP + 1

		N = len(clusters)
		Ngt = len(total_clusters)
		R = N/Ngt
		P = TP/X
		F1 = 2 * ((P*R)/(P+R))

		return R, P, F1
		#print("Total Number of clusters: ", total_clusters, Ngt)
		#print("Clusters in images: ", clusters, N)
		#print("Cluster Recall: ", R)
		#print("TP: ", TP)	
		#print("Precision: ", P)
		#print("F1-score: ", F1)

	def attributes_filter(self, counts):

		verbs = {}
		nouns = {}
		for word in counts:
			lista = word2lemma_pos(word)
			if len(lista) == 1:
				attribute = lista[0]
				if attribute[1] == "VERB":
					verbs[attribute[0]] = counts[word]
				if attribute[1] == "NOUN":
					nouns[attribute[0]] = counts[word]

		return verbs, nouns

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
				##if we wanto to filter more, we can treshold the similarity (sim_score and con_score)
				sim_score = similarity(w_lemma[0], con)
				if sim_score <= 0:
					sim_score = 0

				if d[con] >= 1:
					con_score = 1*sim_score
				else:
					con_score = d[con]*sim_score
				#print(w_lemma[0], con, con_score)

				if con_score >= 0:
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



