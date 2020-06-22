from django.shortcuts import render
from django.views import View
from django.http import JsonResponse

from fileupload.models import (ImageModel, LocationModel, ConceptModel, ConceptScoreModel, 
								CategoryModel, CategoryScoreModel, ActivityInfoModel, ActivityModel, 
								AttributesModel, AttributesInfoModel)
from django.core import serializers

from .models import TopicModel, SimilarityModel

from .auxiliary import create_dict, compute_score, best_clusters, evaluation, retrieve_images
from .sentence_analyzer.nlp_analyzer import similarity, word2lemma

from tqdm import tqdm
import numpy as np

import os
from django.conf import settings
from django.db.models import Q

import time
import collections
import itertools 
import datetime

from sklearn.cluster import DBSCAN

class LMRT_TestView(View):

	template_name = "retrieval/lmrt_test.html"

	def computeSimilarity(self, item, query_word):

		sim_objs = SimilarityModel.objects.filter(Q(word1=item, word2=query_word.tag) | Q(word1=query_word.tag, word2=item))
		sim_score=0

		if (sim_objs):
			for so in sim_objs:
				sim_score = so.score
		else:
			sim_score = similarity(item, query_word.tag)
			SimilarityModel.objects.create(word1=item, word2=query_word.tag, score=sim_score)

		return sim_score

	def convert2lemma(self, words):
		lemmas = []
		for item in words:
			lemmas.append(word2lemma(item))
		return lemmas

	def getWordFilter(self, imgsRetrieval, words, query_word, datefilter, search_mode):

		for item in words:

			sim_score = self.computeSimilarity(item, query_word)

			if sim_score >= 0.6:
				print(item, query_word.tag, sim_score)
				if isinstance(query_word, ConceptModel) and (search_mode in ["objects"]):
					tmpFilter = datefilter & Q(concepts__tag__contains=query_word.tag)
					tmp = ImageModel.objects.filter(tmpFilter)
					print("Number Concepts Images: ",tmp.count())
					for img in tmp:

						tmpScores = ConceptScoreModel.objects.filter(image=img, tag=query_word)
						if tmpScores:
							if img.slug not in imgsRetrieval:
								imgsRetrieval[img.slug] = {}
							if search_mode not in imgsRetrieval[img.slug]:
								imgsRetrieval[img.slug][search_mode] = {}
							if item not in imgsRetrieval[img.slug][search_mode]:
								imgsRetrieval[img.slug][search_mode][item] = {}
							if query_word.tag not in imgsRetrieval[img.slug][search_mode][item]:
								imgsRetrieval[img.slug][search_mode][item][query_word.tag] = 0
							for sc in tmpScores:
								labelScore = sc.score*0.5 + sim_score*0.5		
								if labelScore > imgsRetrieval[img.slug][search_mode][item][query_word.tag]:
									imgsRetrieval[img.slug][search_mode][item][query_word.tag] = labelScore

				if isinstance(query_word, CategoryModel) and (search_mode in ["locations", "objects"]):
					tmpFilter = datefilter & Q(categories__tag__contains=query_word.tag)
					tmp = ImageModel.objects.filter(tmpFilter)
					#print("Number: ",tmp.count())
					print("Number Categories Images: ",tmp.count())
					for img in tmp:

						tmpScores = CategoryScoreModel.objects.filter(image=img, tag=query_word)
						if tmpScores:
							if img.slug not in imgsRetrieval:
								imgsRetrieval[img.slug] = {}
							if search_mode not in imgsRetrieval[img.slug]:
								imgsRetrieval[img.slug][search_mode] = {}
							if item not in imgsRetrieval[img.slug][search_mode]:
								imgsRetrieval[img.slug][search_mode][item] = {}
							if query_word.tag not in imgsRetrieval[img.slug][search_mode][item]:
								imgsRetrieval[img.slug][search_mode][item][query_word.tag] = 0
							for sc in tmpScores:
								labelScore = sc.score*0.3 + sim_score*0.7							
								if labelScore > imgsRetrieval[img.slug][search_mode][item][query_word.tag]:
									imgsRetrieval[img.slug][search_mode][item][query_word.tag] = labelScore

				if isinstance(query_word, ActivityModel) and (search_mode in ["activities"]):
					tmpFilter = datefilter & Q(activities__tag__contains=query_word.tag)
					tmp = ImageModel.objects.filter(tmpFilter)
					print("Number Activities Images: ",tmp.count())
					#print("Number: ",tmp.count())
					for img in tmp:
						if img.slug not in imgsRetrieval:
							imgsRetrieval[img.slug] = {}
						if search_mode not in imgsRetrieval[img.slug]:
							imgsRetrieval[img.slug][search_mode] = {}
						if item not in imgsRetrieval[img.slug][search_mode]:
							imgsRetrieval[img.slug][search_mode][item] = {}
						if query_word.tag not in imgsRetrieval[img.slug][search_mode][item]:
							imgsRetrieval[img.slug][search_mode][item][query_word.tag] = 0
						
						if sim_score > imgsRetrieval[img.slug][search_mode][item][query_word.tag]:
							imgsRetrieval[img.slug][search_mode][item][query_word.tag] = sim_score


				if isinstance(query_word, LocationModel) and (search_mode in ["locations"]):
					tmpFilter = datefilter & Q(location__tag__contains=query_word.tag)
					tmp = ImageModel.objects.filter(tmpFilter)
					print("Number Locations Images: ",tmp.count())
					#print("Number: ",tmp.count())
					for img in tmp:
						if img.slug not in imgsRetrieval:
							imgsRetrieval[img.slug] = {}
						if search_mode not in imgsRetrieval[img.slug]:
							imgsRetrieval[img.slug][search_mode] = {}
						if item not in imgsRetrieval[img.slug][search_mode]:
							imgsRetrieval[img.slug][search_mode][item] = {}
						if query_word.tag not in imgsRetrieval[img.slug][search_mode][item]:
							imgsRetrieval[img.slug][search_mode][item][query_word.tag] = 0
						
						if sim_score > imgsRetrieval[img.slug][search_mode][item][query_word.tag]:
							imgsRetrieval[img.slug][search_mode][item][query_word.tag] = sim_score

				if isinstance(query_word, AttributesModel) and (search_mode in ["locations", "activities", "objects"]):
					tmpFilter = datefilter & Q(attributes__tag__contains=query_word.tag)
					tmp = ImageModel.objects.filter(tmpFilter)
					print("Number Attributes Images: ",tmp.count())
					#print("Number: ",tmp.count())
					for img in tmp:
						if img.slug not in imgsRetrieval:
							imgsRetrieval[img.slug] = {}
						if search_mode not in imgsRetrieval[img.slug]:
							imgsRetrieval[img.slug][search_mode] = {}
						if item not in imgsRetrieval[img.slug][search_mode]:
							imgsRetrieval[img.slug][search_mode][item] = {}
						if query_word.tag not in imgsRetrieval[img.slug][search_mode][item]:
							imgsRetrieval[img.slug][search_mode][item][query_word.tag] = 0
						
						if sim_score > imgsRetrieval[img.slug][search_mode][item][query_word.tag]:
							imgsRetrieval[img.slug][search_mode][item][query_word.tag] = sim_score


	def isNegative(self, words, query_word):

		for item in words:

			sim_score = self.computeSimilarity(item, query_word)
			if sim_score >= 0.7:
				return True

		return False

	def computeImageScore(self, data, search_labels):

		#print(data)
		#imgsRetrieval[img.slug][search_mode][item][query_word.tag] = sim_score
		imgScore = 0
		for s in search_labels:
			#print(s)
			if s in data:
				tmpScore = 0
				for k in data[s]:
					#print(data[s][k])
					for j in data[s][k]:
						if tmpScore < data[s][k][j]:
							tmpScore = data[s][k][j]
				
					#print(tmpScore)	
				imgScore = imgScore + tmpScore

		imgScore = imgScore / len(search_labels)

		return round(imgScore, 4)

	def computeClusters(self, img_clusters):

		features = []

		for img, conf in img_clusters:
			features.append(datetime.datetime.timestamp(img.date_time)/1000000000)
			#print(datetime.datetime.timestamp(img.date_time)/1000000000)

		X = np.asarray(features).reshape(-1, 1)

		#print(X)
		db = DBSCAN(eps=0.0000036, min_samples=2).fit(X)
		labels = db.labels_

		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		n_noise_ = list(labels).count(-1)

		#print(n_clusters_, n_noise_)
		cluster_n = None

		d = {}
		#image_list[img] = [{'url': url, 'conf': score}]
		for item, cl in zip(img_clusters, labels):
			#print(item[0].slug, item[1], cl)
			n_cl = str(cl)
			
			if n_cl not in d:
				d[n_cl] = [{'url': item[0].file.url, 'conf': item[1], 'name': item[0].slug}]
			else:
				d[n_cl].append({'url': item[0].file.url, 'conf': item[1], 'name': item[0].slug})

		#print(d)
		return d
				
	def post(self, request, *args, **kwargs):

		imgsRetrieval = {}
		search_labels = []

		if request.is_ajax():
			image_list = {}

			objects = self.convert2lemma(request.POST.getlist('obj_tags[]'))
			if objects:
				search_labels.append("objects")
			#print(objects)
			activities = self.convert2lemma(request.POST.getlist('act_tags[]'))
			if activities:
				search_labels.append("activities")
			#print(activities)
			locations = self.convert2lemma(request.POST.getlist('loc_tags[]'))
			if locations:
				search_labels.append("locations")
			#print(locations)
			negatives = self.convert2lemma(request.POST.getlist('neg_tags[]'))
			#print(negatives)
			daterange = request.POST.getlist('daterange')
			#print(daterange)
			years = request.POST.getlist('years[]')
			#print(years)
			daysweek = request.POST.getlist('daysweek[]')
			#print(daysweek)

			dateFilter = Q()

			#Filtering images based on dates
			if( daterange or years or daysweek ):
				date_filter = Q()
				if daterange:
					dates = daterange[0].split(' - ')
					start_date = datetime.datetime.strptime(dates[0], '%d/%m/%Y').strftime('%Y-%m-%d')
					end_date =  datetime.datetime.strptime(dates[1], '%d/%m/%Y')
					new_end = end_date + datetime.timedelta(days=1)

					date_filter = date_filter & Q(date_time__range=[start_date, new_end.strftime('%Y-%m-%d')])
				
				if years:
					year_filter = Q()
					for y in years:
						year_filter = year_filter | Q(date_time__year=y)

					date_filter = date_filter & year_filter

				if daysweek:
					day_filter = Q()
					for day in daysweek:
						day_filter = day_filter | Q(date_time__week_day=int(day))

					date_filter = date_filter & day_filter

				dateFilter = date_filter
			#print(final_filter)

			#Filtering images based on labels and annotations
			query_concepts = ConceptModel.objects.all()
			#print(query_concepts)
			query_categories = CategoryModel.objects.all()
			#print(query_categories)
			query_activities = ActivityModel.objects.all()
			#print(query_activities)
			query_attributes = AttributesModel.objects.all()
			#print(query_attributes)
			query_locations = LocationModel.objects.all()
			#print(query_locations)

			all_queries = list(query_concepts) + list(query_categories) + list(query_activities) + list(query_attributes) + list(query_locations)
			#print(all_queries)


			for item in tqdm(all_queries):

				#Negative (Irrelevant) words are obtained to create a filter
				#nFilter = nFilter | self.getWordFilter(imgRetrieval, negatives, item, dateFilter)
				if not self.isNegative(negatives, item):
					#Filtering objects
					self.getWordFilter(imgsRetrieval, objects, item, dateFilter, "objects")
					#Filtering activities
					self.getWordFilter(imgsRetrieval, activities, item, dateFilter, "activities")
					#Filtering locations
					self.getWordFilter(imgsRetrieval, locations, item, dateFilter, "locations")

			
			print(len(imgsRetrieval))

			img_clusters = []
			for img in tqdm(imgsRetrieval):

				img_obj = ImageModel.objects.filter(slug=img)[0]
				#url = img_obj.file.url
				
				score = self.computeImageScore(imgsRetrieval[img], search_labels)
				if score > 0.6:
					#print(img, score) 
					#image_list[img] = [{'url': url, 'conf': score}]
					img_clusters.append((img_obj, score))


			print(len(img_clusters))

			#Creating images clustering using temporal data
			if img_clusters:
				image_list = self.computeClusters(img_clusters)

			print(len(image_list))

			return JsonResponse({"success": True, "queryset": image_list}, status=200)
		else:
			return JsonResponse({"success": False}, status=400)


	def get(self, request, *args, **kwargs):

		context = {}
		return render(self.request, self.template_name, context)


class LMRT_TestView_v1(View):

	template_name = "retrieval/lmrt_test_v1.html"

	def convert2lemma(self, words):
		lemmas = []
		for item in words:
			lemmas.append(word2lemma(item))
		return lemmas

	def getWordFilter(self, words, query_word, search_mode=None):

		f = Q()
		for item in words:

			sim_objs = SimilarityModel.objects.filter(Q(word1=item, word2=query_word.tag) | Q(word1=query_word.tag, word2=item))
			sim_score=0

			if (sim_objs):
				for so in sim_objs:
					sim_score = so.score
			else:
				sim_score = similarity(item, query_word.tag)
				SimilarityModel.objects.create(word1=item, word2=query_word.tag, score=sim_score)

			if sim_score >= 0.65:
				#print(item, query_word.tag, sim_score)
				if isinstance(query_word, ConceptModel) and (search_mode in ["objects", None]):
					f = f | Q(concepts__tag__contains=query_word.tag)
				if isinstance(query_word, CategoryModel) and (search_mode in ["locations", "objects", None]):
					f = f | Q(categories__tag__contains=query_word.tag)
				if isinstance(query_word, ActivityModel) and (search_mode in ["activities", None]):
					f = f | Q(activities__tag__contains=query_word.tag)
				if isinstance(query_word, LocationModel) and (search_mode in ["locations", None]):
					f = f | Q(location__tag__contains=query_word.tag)
				if isinstance(query_word, AttributesModel) and (search_mode in ["locations", "activities", "objects", None]):
					f = f | Q(attributes__tag__contains=query_word.tag)

		return f

				
	def post(self, request, *args, **kwargs):

		if request.is_ajax():
			image_list = {}

			objects = self.convert2lemma(request.POST.getlist('obj_tags[]'))
			#print(objects)
			activities = self.convert2lemma(request.POST.getlist('act_tags[]'))
			#print(activities)
			locations = self.convert2lemma(request.POST.getlist('loc_tags[]'))
			#print(locations)
			negatives = self.convert2lemma(request.POST.getlist('neg_tags[]'))
			#print(negatives)
			daterange = request.POST.getlist('daterange')
			#print(daterange)
			years = request.POST.getlist('years[]')
			#print(years)
			daysweek = request.POST.getlist('daysweek[]')
			#print(daysweek)

			final_filter = Q()

			#Filtering images based on dates
			if( daterange or years or daysweek ):
				date_filter = Q()
				if daterange:
					dates = daterange[0].split(' - ')
					start_date = datetime.datetime.strptime(dates[0], '%d/%m/%Y').strftime('%Y-%m-%d')
					end_date =  datetime.datetime.strptime(dates[1], '%d/%m/%Y')
					new_end = end_date + datetime.timedelta(days=1)

					print(start_date, new_end)

					date_filter = date_filter & Q(date_time__range=[start_date, new_end.strftime('%Y-%m-%d')])
				
				if years:
					year_filter = Q()
					for y in years:
						year_filter = year_filter | Q(date_time__year=y)

					date_filter = date_filter & year_filter

				if daysweek:
					day_filter = Q()
					for day in daysweek:
						day_filter = day_filter | Q(date_time__week_day=int(day))

					date_filter = date_filter & day_filter

				final_filter = date_filter
			#print(final_filter)

			#Filtering images based on labels and annotations

			query_concepts = ConceptModel.objects.all()
			#print(query_concepts)
			query_categories = CategoryModel.objects.all()
			#print(query_categories)
			query_activities = ActivityModel.objects.all()
			#print(query_activities)
			query_attributes = AttributesModel.objects.all()
			#print(query_attributes)
			query_locations = LocationModel.objects.all()
			#print(query_locations)

			all_queries = list(query_concepts) + list(query_categories) + list(query_activities) + list(query_attributes) + list(query_locations)
			#print(all_queries)

			nFilter = Q()
			objFilter = Q()
			actFilter = Q()
			locFilter = Q()

			for item in tqdm(all_queries):

				#Negative (Irrelevant) words are obtained to create a filter
				nFilter = nFilter | self.getWordFilter(negatives, item)

				#Filtering objects
				objFilter = objFilter | self.getWordFilter(objects, item, "objects")

				#Filtering activities
				actFilter = actFilter | self.getWordFilter(activities, item, "activities")

				#Filtering locations
				locFilter = locFilter | self.getWordFilter(locations, item, "locations")


			final_filter = final_filter & objFilter & actFilter & locFilter & ~nFilter
			print(final_filter)

			final_set = ImageModel.objects.filter(final_filter).distinct()
			print(final_set.count())

			for img in final_set:
				name = img.slug
				url =  img.file.url
				score = 100
				image_list[name] = [{'url': url, 'conf': score}]

			return JsonResponse({"success": True, "queryset": image_list}, status=200)
		else:
			return JsonResponse({"success": False}, status=400)


	def get(self, request, *args, **kwargs):

		context = {}
		return render(self.request, self.template_name, context)


class LMRTConnectionsView(View):

	template_name = "retrieval/lmrt.html"

	def post(self, request, *args, **kwargs):

		if request.is_ajax():

			tic = time.clock()

			search_words = {}
			image_list = {}
			evaluation_data = {}

			objects = request.POST.getlist('obj_tags[]')
			#if objects:
				#search_words["objects"] = objects

			activities = request.POST.getlist('act_tags[]')
			#if activities:
				#search_words["activities"] = activities

			locations = request.POST.getlist('loc_tags[]')
			#if locations:
				#search_words["locations"] = locations

			negatives = request.POST.getlist('neg_tags[]')
			daterange = request.POST.getlist('daterange')
			years = request.POST.getlist('years[]')
			daysweek = request.POST.getlist('daysweek[]')
			topic_id = request.POST.getlist('topic_id')[0]

			#print(search_words)
			final_filter = Q()
			if( daterange or years or daysweek ):

				date_filter = Q()

				if daterange:
					dates = daterange[0].split(' - ')
					start_date = datetime.datetime.strptime(dates[0], '%d/%m/%Y').strftime('%Y-%m-%d')
					end_date =  datetime.datetime.strptime(dates[1], '%d/%m/%Y')
					new_end = end_date + datetime.timedelta(days=1)

					date_filter = date_filter & Q(date_time__range=[start_date, new_end.strftime('%Y-%m-%d')])
				
				if years:
					year_filter = Q()
					for y in years:
						year_filter = year_filter | Q(date_time__year=y)

					date_filter = date_filter & year_filter

				if daysweek:
					day_filter = Q()
					for day in daysweek:
						day_filter = day_filter | Q(date_time__week_day=int(day))

					date_filter = date_filter & day_filter

				final_filter = date_filter

			#print(final_filter)

			query_concepts = ConceptModel.objects.all()
			query_categories = CategoryModel.objects.all()
			query_activities = ActivityModel.objects.all()
			query_attributes = AttributesModel.objects.all()
			query_locations = LocationModel.objects.all()

			neg_filter = Q()
			if negatives:
				#queryset = list(query_concepts) + list(query_categories) + list(query_activities) + list(query_attributes) + list(query_locations)
				
				for word in tqdm(negatives):
					print("Word: ", word)

					w_lemma = word2lemma(word)	
					
					#print("Concepts: ")
					con_filter = Q()
					for con_obj in query_concepts:
						con_lemma = con_obj.tag

						f = Q(word1=w_lemma, word2=con_lemma) | Q(word1=con_lemma, word2=w_lemma)
						sim_objs = SimilarityModel.objects.filter(f)
						sim_score=0
						if (sim_objs):
							for so in sim_objs:
								sim_score = so.score
						else:
							sim_score = similarity(w_lemma, con_lemma)
							SimilarityModel.objects.create(word1=w_lemma, word2=con_lemma, score=sim_score)

						if sim_score >= 0.65:
							#print(con_lemma, w_lemma, sim_score)
							con_filter = con_filter | Q(concepts__tag__contains=con_obj) #| Q(location__tag__contains=con_obj) | Q(categories__tag__contains=con_obj) | Q(activities__tag__contains=con_obj) | Q(attributes__tag__contains=con_obj)
					
					#print(con_filter)

					#print("Categories: ")
					cat_filter = Q()
					for con_obj in query_categories:
						con_lemma = con_obj.tag

						f = Q(word1=w_lemma, word2=con_lemma) | Q(word1=con_lemma, word2=w_lemma)
						sim_objs = SimilarityModel.objects.filter(f)
						sim_score=0
						if (sim_objs):
							for so in sim_objs:
								sim_score = so.score
						else:
							sim_score = similarity(w_lemma, con_lemma)
							SimilarityModel.objects.create(word1=w_lemma, word2=con_lemma, score=sim_score)

						if sim_score >= 0.65:
							#print(con_lemma, w_lemma, sim_score)
							cat_filter = cat_filter | Q(categories__tag__contains=con_obj)

					#print(cat_filter)

					#print("Activities: ")
					act_filter = Q()
					for con_obj in query_activities:
						con_lemma = con_obj.tag

						f = Q(word1=w_lemma, word2=con_lemma) | Q(word1=con_lemma, word2=w_lemma)
						sim_objs = SimilarityModel.objects.filter(f)
						sim_score=0
						if (sim_objs):
							for so in sim_objs:
								sim_score = so.score
						else:
							sim_score = similarity(w_lemma, con_lemma)
							SimilarityModel.objects.create(word1=w_lemma, word2=con_lemma, score=sim_score)

						if sim_score >= 0.65:
							#print(con_lemma, w_lemma, sim_score)
							act_filter = act_filter | Q(activities__tag__contains=con_obj)

					#print(act_filter)

					#print("Attributes: ")
					att_filter = Q()
					for con_obj in query_attributes:
						con_lemma = con_obj.tag

						f = Q(word1=w_lemma, word2=con_lemma) | Q(word1=con_lemma, word2=w_lemma)
						sim_objs = SimilarityModel.objects.filter(f)
						sim_score=0
						if (sim_objs):
							for so in sim_objs:
								sim_score = so.score
						else:
							sim_score = similarity(w_lemma, con_lemma)
							SimilarityModel.objects.create(word1=w_lemma, word2=con_lemma, score=sim_score)

						if sim_score >= 0.65:
							#print(con_lemma, w_lemma, sim_score)
							att_filter = att_filter | Q(attributes__tag__contains=con_obj)

					#print(att_filter)

					#print("Locations: ")
					loc_filter = Q()
					for con_obj in query_locations:
						con_lemma = con_obj.tag

						#print(con_lemma)

						f = Q(word1=w_lemma, word2=con_lemma) | Q(word1=con_lemma, word2=w_lemma)
						sim_objs = SimilarityModel.objects.filter(f)
						sim_score=0
						if (sim_objs):
							for so in sim_objs:
								sim_score = so.score
						else:
							sim_score = similarity(w_lemma, con_lemma)
							SimilarityModel.objects.create(word1=w_lemma, word2=con_lemma, score=sim_score)

						if sim_score >= 0.65:
							#print(con_lemma, w_lemma, sim_score)
							loc_filter = loc_filter | Q(location__tag__contains=con_obj)

					#print(loc_filter)

					neg_filter = neg_filter | con_filter | cat_filter | act_filter | att_filter | loc_filter

				neg_set = ImageModel.objects.filter(neg_filter).distinct()
				print(neg_set.count())


			obj_filter = Q()
			if objects:
				#queryset = list(query_concepts) + list(query_categories) + list(query_activities) + list(query_attributes) + list(query_locations)
				
				for word in tqdm(objects):
					print("Word: ", word)

					w_lemma = word2lemma(word)	
					
					#print("Concepts: ")
					con_filter = Q()
					for con_obj in query_concepts:
						con_lemma = con_obj.tag

						f = Q(word1=w_lemma, word2=con_lemma) | Q(word1=con_lemma, word2=w_lemma)
						sim_objs = SimilarityModel.objects.filter(f)
						sim_score=0
						if (sim_objs):
							for so in sim_objs:
								sim_score = so.score
						else:
							sim_score = similarity(w_lemma, con_lemma)
							SimilarityModel.objects.create(word1=w_lemma, word2=con_lemma, score=sim_score)

						if sim_score >= 0.65:
							#print(con_lemma, w_lemma, sim_score)
							#search_words["objects"][word] = { "tag": con_lemma, "sim_score": sim_score }
							con_filter = con_filter | Q(concepts__tag__contains=con_obj) #| Q(location__tag__contains=con_obj) | Q(categories__tag__contains=con_obj) | Q(activities__tag__contains=con_obj) | Q(attributes__tag__contains=con_obj)
					#print(con_filter)
					obj_filter = obj_filter | con_filter

				#print(obj_filter)

				if len(obj_filter) > 0:
					obj_set = ImageModel.objects.filter(obj_filter).distinct()
					print(obj_set.count())


			c_filter = Q()
			if locations:

				for word in tqdm(locations):
					print("Word: ", word)

					w_lemma = word2lemma(word)

					cat_filter = Q()
					for con_obj in query_categories:
						con_lemma = con_obj.tag

						f = Q(word1=w_lemma, word2=con_lemma) | Q(word1=con_lemma, word2=w_lemma)
						sim_objs = SimilarityModel.objects.filter(f)
						sim_score=0
						if (sim_objs):
							for so in sim_objs:
								sim_score = so.score
						else:
							sim_score = similarity(w_lemma, con_lemma)
							SimilarityModel.objects.create(word1=w_lemma, word2=con_lemma, score=sim_score)

						if sim_score >= 0.65:
							#print(con_lemma, w_lemma, sim_score)
							cat_filter = cat_filter | Q(categories__tag__contains=con_obj)


					loc_filter = Q()
					for con_obj in query_locations:
						con_lemma = con_obj.tag

						#print(con_lemma)

						f = Q(word1=w_lemma, word2=con_lemma) | Q(word1=con_lemma, word2=w_lemma)
						sim_objs = SimilarityModel.objects.filter(f)
						sim_score=0
						if (sim_objs):
							for so in sim_objs:
								sim_score = so.score
						else:
							sim_score = similarity(w_lemma, con_lemma)
							SimilarityModel.objects.create(word1=w_lemma, word2=con_lemma, score=sim_score)

						if sim_score >= 0.65:
							#print(con_lemma, w_lemma, sim_score)
							loc_filter = loc_filter | Q(location__tag__contains=con_obj)


					c_filter = c_filter | cat_filter | loc_filter



				#print(c_filter)
				cat_set = ImageModel.objects.filter(c_filter).distinct()
				print(cat_set.count())


			a_filter = Q()
			if activities:
				for word in tqdm(activities):
					print("Word: ", word)

					w_lemma = word2lemma(word)

					#print("Activities: ")
					act_filter = Q()
					for con_obj in query_activities:
						con_lemma = con_obj.tag

						f = Q(word1=w_lemma, word2=con_lemma) | Q(word1=con_lemma, word2=w_lemma)
						sim_objs = SimilarityModel.objects.filter(f)
						sim_score=0
						if (sim_objs):
							for so in sim_objs:
								sim_score = so.score
						else:
							sim_score = similarity(w_lemma, con_lemma)
							SimilarityModel.objects.create(word1=w_lemma, word2=con_lemma, score=sim_score)

						if sim_score >= 0.65:
							print(con_lemma, w_lemma, sim_score)
							act_filter = act_filter | Q(activities__tag__contains=con_obj)

					print(act_filter)

					#print("Attributes: ")
					att_filter = Q()
					for con_obj in query_attributes:
						con_lemma = con_obj.tag

						f = Q(word1=w_lemma, word2=con_lemma) | Q(word1=con_lemma, word2=w_lemma)
						sim_objs = SimilarityModel.objects.filter(f)
						sim_score=0
						if (sim_objs):
							for so in sim_objs:
								sim_score = so.score
						else:
							sim_score = similarity(w_lemma, con_lemma)
							SimilarityModel.objects.create(word1=w_lemma, word2=con_lemma, score=sim_score)

						if sim_score >= 0.65:
							print(con_lemma, w_lemma, sim_score)
							att_filter = att_filter | Q(attributes__tag__contains=con_obj)
					print(att_filter)
					a_filter = a_filter | act_filter | att_filter

				print(a_filter)
				act_set = ImageModel.objects.filter(a_filter).distinct()
				print(act_set.count())

			
			final_filter = final_filter & obj_filter & c_filter & a_filter & ~neg_filter
			#print(final_filter)
			final_set = ImageModel.objects.filter(final_filter).distinct()
			print(final_set.count())

			evaluation_list = {}
			max_conf = 0
			img_clusters = []
			for img in tqdm(final_set):

				scores = []

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

				if max_conf < img_conf:
					max_conf = img_conf
				
				if img_conf > (max_conf*0.5) and img_conf >= 0.2:	
					evaluation_list[img.slug] = { "image": img, "confidence": img_conf }
					img_clusters.append((img, img_conf))
				
			evaluation_data = {}

			image_list = {}
			noise_list = []
			counter = 0
			if img_clusters:
			
				clusters = best_clusters(img_clusters)

				clusters = sorted(clusters.items(), key = lambda item: item[1]['confidence'], reverse=True)
				#print(clusters)
				for c in clusters:
					if c[0] != -1 and c[1]["confidence"] >= (max_conf*0.65):
						name = c[1]["image"].slug
						url =  c[1]["image"].file.url
						score = c[1]["confidence"]*100
						image_list[name] = [{'url': url, 'conf': score}]
						counter = counter + 1
					else:
						noise_list.append(c[1]["image"].slug)

			#print(image_list)

			if evaluation_list:
				img_list_sorted = sorted(evaluation_list.items(), key = lambda item: item[1]['confidence'], reverse=True)
				
				for item in img_list_sorted:

					if (item[0] not in image_list) and (item[0] not in noise_list):
						name = item[1]["image"].slug
						url =  item[1]["image"].file.url
						score = item[1]["confidence"]*100
						image_list[name] = [{'url': url, 'conf': score}]
						counter = counter + 1
						if counter > 299: 
							break
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

			#for image in final_set:
				#image_list[image.slug] = [{'url': image.file.url, 'conf': 1}]
			# objects = request.POST.getlist('obj_tags[]')
			# locations = request.POST.getlist('loc_tags[]')
			# activities = request.POST.getlist('act_tags[]')
			# negatives = request.POST.getlist('neg_tags[]')
			# topic_id = request.POST.getlist('topic_id')[0]
			# daterange = request.POST.getlist('daterange')
			# years = request.POST.getlist('years[]')
			# daysweek = request.POST.getlist('daysweek[]')

			# image_list = {}
			# # evaluation_list = {}
			# evaluation_data = {}
			# images_by_date = ImageModel.objects.none()

			# img_clusters = []

			# max_conf = 0

			# if( objects or locations or activities or negatives):

			# 	if( daterange or years or daysweek ):

			# 		filtro = Q()

			# 		if daterange:
			# 			dates = daterange[0].split(' - ')
			# 			start_date = datetime.datetime.strptime(dates[0], '%d/%m/%Y').strftime('%Y-%m-%d')
			# 			end_date =  datetime.datetime.strptime(dates[0], '%d/%m/%Y')
			# 			new_end = end_date + datetime.timedelta(days=1)

			# 			filtro = filtro | Q(date_time__range=[start_date, new_end.strftime('%Y-%m-%d')])
					
			# 		if years:
			# 			for y in years:
			# 				filtro = filtro | Q(date_time__year=y)

			# 		if daysweek:
			# 			for day in daysweek:
			# 				filtro = filtro | Q(date_time__week_day=int(day))
			# 			#print("Days of Week: ", daysweek)

			# 		images_by_date = ImageModel.objects.filter(filtro)
			# 	else:
			# 		images_by_date = ImageModel.objects.all()

			# 	query_concepts = ConceptModel.objects.all()
			# 	query_categories = CategoryModel.objects.all()
			# 	query_activities = ActivityModel.objects.all()
			# 	query_attributes = AttributesModel.objects.all()
			# 	query_locations = LocationModel.objects.all()

			# 	imgs_neg = {}
			# 	if negatives:
			# 		queryset = list(query_concepts) + list(query_categories) + list(query_activities) + list(query_attributes) #+ list(query_locations)
			# 		imgs_neg = retrieve_images(queryset, negatives)

			# 	imgs_objs = {}
			# 	if objects:
			# 		queryset = list(query_concepts)
			# 		imgs_objs = retrieve_images(queryset, objects)
			# 		#print(imgs_objs)

			# 	imgs_loc = {}
			# 	if locations:
			# 		queryset = list(query_categories) + list(query_locations)
			# 		imgs_loc =  retrieve_images(queryset, locations)

			# 	imgs_act = {}
			# 	if activities:
			# 		queryset =list(query_activities) + list(query_attributes) 
			# 		imgs_act =  retrieve_images(queryset, activities)
			# 		#print(imgs_act)

			# 	selected = list(imgs_objs.keys() | imgs_loc.keys() | imgs_act.keys())

			# 	#print(sorted(selected))

			# 	for img_name in selected:
			# 		s = []
					
			# 		neg_s = None
			# 		try:
			# 			neg_s = imgs_neg[img_name]
			# 		except:
			# 			neg_s = 0

			# 		if neg_s < 0.6:

			# 			if objects:
			# 				try:
			# 					s.append(imgs_objs[img_name])
			# 				except:
			# 					s.append(0)

			# 			if locations:
			# 				try:
			# 					s.append(imgs_loc[img_name])
			# 				except:
			# 					s.append(0)

			# 			if activities:
			# 				try:
			# 					s.append(imgs_act[img_name])
			# 				except:
			# 					s.append(0)

			# 			img_conf = 0
			# 			for ss in s:
			# 				img_conf = img_conf + ss

			# 			if len(s) > 0:
			# 				img_conf = img_conf / len(s)

			# 			img = ImageModel.objects.get(slug=img_name)
			# 			if img_conf > max_conf:
			# 					max_conf=img_conf
			# 			if img_conf > (max_conf*0.5) and img_conf >= 0.2 and (img in images_by_date):
			# 				evaluation_list[img_name] = { "image": img, "confidence": img_conf }
			# 				img_clusters.append((img, img_conf))

			# 	noise_list = []
			# 	counter = 0
			# 	if img_clusters:
				
			# 		clusters = best_clusters(img_clusters)

			# 		clusters = sorted(clusters.items(), key = lambda item: item[1]['confidence'], reverse=True)

			# 		#print(clusters)

			# 		for c in clusters:
			# 			if c[0] != -1 and c[1]["confidence"] >= (max_conf*0.95):
			# 				name = c[1]["image"].slug
			# 				url =  c[1]["image"].file.url
			# 				score = c[1]["confidence"]*100
			# 				image_list[name] = [{'url': url, 'conf': score}]
			# 				counter = counter + 1
			# 			else:
			# 				noise_list.append(c[1]["image"].slug)

			# 	#print(image_list)

			# 	if evaluation_list:
			# 		img_list_sorted = sorted(evaluation_list.items(), key = lambda item: item[1]['confidence'], reverse=True)
					
			# 		for item in img_list_sorted:

			# 			if (item[0] not in image_list) and (item[0] not in noise_list):
			# 				name = item[1]["image"].slug
			# 				url =  item[1]["image"].file.url
			# 				score = item[1]["confidence"]*100
			# 				image_list[name] = [{'url': url, 'conf': score}]
			# 				counter = counter + 1
			# 				if counter > 50: 
			# 					break
			# 				#print(item[0], image_list[item[0]])

			# 		#print(img_list_sorted)

			# 		recall, precision, f1_score = evaluation(dict(itertools.islice(image_list.items(), 5)), topic_id)
			# 		evaluation_data["Top5"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]
			# 		recall, precision, f1_score = evaluation(dict(itertools.islice(image_list.items(), 10)), topic_id)
			# 		evaluation_data["Top10"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]
			# 		recall, precision, f1_score = evaluation(dict(itertools.islice(image_list.items(), 20)), topic_id)
			# 		evaluation_data["Top20"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]
			# 		recall, precision, f1_score = evaluation(dict(itertools.islice(image_list.items(), 30)), topic_id)
			# 		evaluation_data["Top30"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]
			# 		recall, precision, f1_score = evaluation(dict(itertools.islice(image_list.items(), 40)), topic_id)
			# 		evaluation_data["Top40"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]
			# 		recall, precision, f1_score = evaluation(dict(itertools.islice(image_list.items(), 50)), topic_id)
			# 		evaluation_data["Top50"] = [{ "recall": recall, "precision": precision, "f1_score": f1_score}]


			toc = time.clock()
			print("Processing time: ", (toc - tic))

			return JsonResponse({"success": True, "queryset": image_list, "evaluation": evaluation_data}, status=200)
		else:
			return JsonResponse({"success": False}, status=400)

	def get(self, request, *args, **kwargs):

		topicset = TopicModel.objects.all()
		context = { 'topics': topicset }
		return render(self.request, self.template_name, context)

class LMRTView(View):

	template_name = "retrieval/lmrt.html"

	def post(self, request, *args, **kwargs):

		if request.is_ajax():

			tic = time.clock()
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
				max_conf = 0
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

					if max_conf < img_conf:
						max_conf = img_conf
					
					if img_conf > (max_conf*0.5) and img_conf >= 0.2:	
						evaluation_list[img.slug] = { "image": img, "confidence": img_conf }
						img_clusters.append((img, img_conf))
					
				evaluation_data = {}

				image_list = {}
				noise_list = []
				counter = 0
				if img_clusters:
				
					clusters = best_clusters(img_clusters)

					clusters = sorted(clusters.items(), key = lambda item: item[1]['confidence'], reverse=True)
					#print(clusters)
					for c in clusters:
						if c[0] != -1 and c[1]["confidence"] >= (max_conf*0.65):
							name = c[1]["image"].slug
							url =  c[1]["image"].file.url
							score = c[1]["confidence"]*100
							image_list[name] = [{'url': url, 'conf': score}]
							counter = counter + 1
						else:
							noise_list.append(c[1]["image"].slug)

				#print(image_list)

				if evaluation_list:
					img_list_sorted = sorted(evaluation_list.items(), key = lambda item: item[1]['confidence'], reverse=True)
					
					for item in img_list_sorted:

						if (item[0] not in image_list) and (item[0] not in noise_list):
							name = item[1]["image"].slug
							url =  item[1]["image"].file.url
							score = item[1]["confidence"]*100
							image_list[name] = [{'url': url, 'conf': score}]
							counter = counter + 1
							if counter > 50: 
								break
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

			toc = time.clock()
			print("Processing time: ", (toc - tic))

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



