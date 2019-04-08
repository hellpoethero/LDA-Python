import random
from LdaModel import LdaModel
from LdaInference import LdaInference
from LdaAlpha import LdaAlpha
from Corpus import Corpus
import numpy as np


class LdaEstimate:
	LAG = 10
	NUM_INIT = 1
	EM_CONVERGED = 0
	EM_MAX_ITER = 0
	ESTIMATE_ALPHA = 0
	INITIAL_ALPHA = 0
	K = 0

	def __init__(self):
		self.a = 1

	@staticmethod
	def myrand():
		return random.random()

	@staticmethod
	def initial_model(start, corpus, num_topics, alpha):
		model = LdaModel()
		if start == "seeded":
			model.set_model(corpus.num_terms, num_topics)
			model.alpha = alpha

			for k in range(0, num_topics):
				for i in range(0, LdaEstimate.NUM_INIT):
					d = int(np.floor(LdaEstimate.myrand() * corpus.num_docs))
					doc = corpus.docs[d]
					for n in range(0, doc.length):
						model.class_word[k][doc.words[n]] += doc.counts[n]
				for n in range(0, model.num_term):
					model.class_word[k][n] += 1/model.num_term
					model.class_total[k] += model.class_word[k][n]
		elif start == "random":
			model.set_model(corpus.num_terms, num_topics)
			model.alpha = alpha
			for k in range(0, num_topics):
				for n in range(0, model.num_term):
					model.class_word[k][n] += 1/model.num_term + LdaEstimate.myrand()
					model.class_total[k] += model.class_word[k][n]
		else:
			model.load_model(start)

		return model

	# calculate gamma and phi
	@staticmethod
	def doc_em(doc, gamma, model, next_model):
		phi = []
		for n in range(0, doc.length):
			temp_array = []
			for k in range(0, model.num_topics):
				temp_array.append(0)
			phi.append(temp_array)

		likelihood = LdaInference.inference(doc, model, gamma, phi)
		for n in range(0, doc.length):
			for k in range(0, model.num_topics):
				next_model.class_word[k][doc.words[n]] += doc.counts[n] * phi[n][k]
				next_model.class_total[k] += doc.counts[n] * phi[n][k]
		# print(likelihood)
		return likelihood

	@staticmethod
	def save_gamma(filename, gamma, num_docs, num_topics):
		pass

	# estimate parameter for model
	@staticmethod
	def run_em(start, directory, corpus):
		likelihood_old = float("-inf")
		print(likelihood_old)
		converged = 1
		var_gamma = []
		for d in range(0, corpus.num_docs):
			temp_array = []
			for k in range(0, LdaEstimate.K):
				temp_array.append(0)
			var_gamma.append(temp_array)

		model = LdaEstimate.initial_model(start, corpus, LdaEstimate.K, LdaEstimate.INITIAL_ALPHA)

		i = 0
		while (converged > LdaEstimate.EM_CONVERGED or i <= 2) and i <= LdaEstimate.EM_MAX_ITER:
			i += 1
			likelihood = 0
			next_model = LdaModel()
			next_model.set_model(model.num_term, model.num_topics)
			next_model.alpha = LdaEstimate.INITIAL_ALPHA
			for d in range(0, corpus.num_docs):
				likelihood += LdaEstimate.doc_em(corpus.docs[d], var_gamma[d], model, next_model)
			if LdaEstimate.ESTIMATE_ALPHA == 1:
				LdaAlpha.maximize_alpha(var_gamma, next_model, corpus.num_docs)

			model = next_model
			converged = (likelihood_old - likelihood) / likelihood
			print(str(i) + " " + str(converged))
			likelihood_old = likelihood

			if i % LdaEstimate.LAG == 0:
				pass

		return model

	@staticmethod
	def read_settings(filename):
		with open(filename, "r") as setting:
			for line in setting:
				fields = line.split(" ")
				if fields[0] == "var_max_iter":
					LdaInference.VAR_MAX_ITER = int(fields[1])
				elif fields[0] == "var_convergence":
					LdaInference.VAR_CONVERGED = float(fields[1])
				elif fields[0] == "em_max_iter":
					LdaEstimate.EM_MAX_ITER = int(fields[1])
				elif fields[0] == "em_convergence":
					LdaEstimate.EM_CONVERGED = float(fields[1])
				elif fields[0] == "alpha":
					if fields[1] == "fixed":
						LdaEstimate.ESTIMATE_ALPHA = 0
					else:
						LdaEstimate.ESTIMATE_ALPHA = 1

	# infer the distribution for new document
	@staticmethod
	def infer(model_root, save, corpus):
		model = LdaModel()
		model.load_model(model_root)
		var_gamma = []
		for d in range(0, corpus.num_docs):
			temp_array = []
			for k in range(0, LdaEstimate.K):
				temp_array.append(0)
			var_gamma.append(temp_array)

		for d in range(0, corpus.num_docs):
			doc = corpus.docs[d]
			phi = []
			for k in range(0, model.num_topics):
				temp_array = []
				for n in range(0, doc.length):
					temp_array.append(0)
				phi.append(temp_array)

			likelihood = LdaInference.inference(doc, model, var_gamma[d], phi)

	@staticmethod
	def estimate(alpha, num_topic, setting, data, vocab, start, output_dir):
		# parameter: est/inf <alpha> <k - num_topic> <setting> <data> <random/seed/*> <output directory>
		LdaEstimate.INITIAL_ALPHA = alpha
		LdaEstimate.K = num_topic
		LdaEstimate.read_settings(setting)
		corpus = Corpus()
		corpus.read(data)

		model = LdaEstimate.run_em(start, output_dir, corpus)
		model.read_vocab(vocab)
		model.print_word_topic_distribution()
		# topic_doc dis and word_topic_dis
