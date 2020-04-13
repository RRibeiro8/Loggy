import numpy as np
import os

import spacy

nlp = spacy.load('en_core_web_md')

def word2lemma(word):

	doc = nlp(word)
	lista = []

	for token in doc:
		lista.append(token.lemma_)

	return lista

def one_word2lemma(word):

	doc = nlp(word)

	lemma = None
	if len(doc) <= 1:
		for token in doc:
			lemma = token.lemma_
	if lemma == None:
		return word
	else:
		return lemma

def similarity(a, b):

	word_a = nlp(a)
	word_b = nlp(b)
	score = word_a.similarity(word_b)
	if score < 0:
		return 0

	return score



if __name__ == "__main__":

	title = nlp("​Icecream by the sea")
	description = nlp("Find the moment when I was eating an ice cream beside the sea.")
	narrative = nlp("​To be relevant, the moment must show both the ice cream with cone in the hand of u1 as well as the sea clearly visible. Any moments by the sea, or eating an ice cream which do not occur together are not considered to be relevant.")

	#for token in description:
		#print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #    token.shape_, token.is_alpha, token.is_stop)

	#for ent in description.ents:
	#	print(ent.text, ent.start_char, ent.end_char, ent.label_)

	#for token in description:
	#	print(token.text, token.has_vector, token.vector_norm, token.is_oov)
	word = nlp("ocean")

	for chunk in description.noun_chunks:
		print(chunk.text, chunk.root.text, chunk.root.dep_,
            chunk.root.head.text)
		print(chunk.text, word.text, chunk.similarity(word))

	for token in description:
		print(token.text, word.text, token.similarity(word))
	