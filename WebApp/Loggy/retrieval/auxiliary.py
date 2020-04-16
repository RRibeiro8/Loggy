import os
from django.conf import settings
from django.db.models import Q

from .models import SimilarityModel

from collections import Counter
from .sentence_analyzer.nlp_analyzer import similarity, word2lemma
import datetime
import numpy as np

from sklearn import preprocessing
from sklearn.cluster import DBSCAN

def best_clusters(img_clusters):

	features = []

	for img, conf in img_clusters:
		features.append(datetime.datetime.timestamp(img.date_time)/1000000000)

	X = np.asarray(features).reshape(-1, 1)

	#print(X)
	db = DBSCAN(eps=0.00003, min_samples=2).fit(X)
	labels = db.labels_

	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)

	#print(n_clusters_, n_noise_)
	cluster_n = None
	best_item = None
	d = {}
	for item, cl in zip(img_clusters, labels):
		#print(item, cl)
		if cluster_n != cl:
			cluster_n = cl
			d[cluster_n] = { "image": item[0], "confidence": item[1]}
		
		if item[1] > d[cluster_n]["confidence"]:
			d[cluster_n] = { "image": item[0], "confidence": item[1]}

	return d

	#url = img.file.url
	#name = img.file.name

	#image_list[name] = [ {'url': url, 'conf': score} ]
	#evaluation_list[img.slug] = img_conf
	#print(X_train)

def scores(word, d):

	w_lemma = word2lemma(word)
	w_score = 0
	for con in d:
		##if we wanto to filter more, we can treshold the similarity (sim_score and con_score)
		if d[con] > 0:
			con_lemma = word2lemma(con)
			f = Q(word1=w_lemma, word2=con_lemma) | Q(word1=con_lemma, word2=w_lemma)
			sim_objs = SimilarityModel.objects.filter(f)
			sim_score=0
			if (sim_objs):
				for so in sim_objs:
					sim_score = so.score
			else:
				sim_score = similarity(w_lemma, con_lemma)
				SimilarityModel.objects.create(word1=w_lemma, word2=con_lemma, score=sim_score)
			
			if sim_score < 0.5:
				con_score = 0
			#elif sim_score >= 0.99 and d[con] >= 0.1:
				#con_score = sim_score
			else:
				con_score = (0.3*d[con]+0.7*sim_score)
			#print(w_lemma[0], con, con_score)

			if con_score > w_score:
				w_score = con_score

	return w_score

def compute_score(objects, d):
	final_objs_score = 0
	final_and_score = 0
	counter_and = 0
	for word in objects:
		if word.startswith("&"):
			word = word.replace("&", "")
			
			w_score = scores(word, d)
			
			final_and_score = final_and_score + w_score
			counter_and = counter_and + 1;

		else:
			w_score = scores(word, d)

			if w_score > final_objs_score:
				final_objs_score = w_score
	
	final_score = 0
	if counter_and > 0:
		final_score = ((final_and_score + final_objs_score) / (counter_and + 1))
	else:
		final_score = final_objs_score
	
	return final_score


def create_dict(img_concepts, img, obj=None):

	d = {}
	for con in img_concepts:
		if obj == None:
			d[con.tag] = 1
		else:

			con_score = 0
			for cs in obj.filter(image=img, tag=con):
				if con_score < cs.score:
					con_score = cs.score

			d[con.tag] = con_score

	return d

def evaluation(img_list, topic_id):

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
					if tmp[2] not in total_clusters:
						total_clusters.append(tmp[2])
					
					for img in img_list:
						#print(img)
						if tmp[1] == img:
							if tmp[2] not in clusters:
								clusters.append(tmp[2])
							#print("positivo", img[0], tmp[2])
							TP = TP + 1

	N = len(clusters)
	Ngt = len(total_clusters)

	R = N/Ngt
	P = TP/X

	F1 = 0
	if (R != 0 or P != 0):
		F1 = 2 * ((P*R)/(P+R))

	return R, P, F1

def attributes_filter(counts):

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

def retrieve_scores(counts, img_at):
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

def word_counter(img_words):

	lista = []
	for c in img_words:
		if c.tag != "NULL":
			lista.append(c.tag)
	return Counter(lista)

