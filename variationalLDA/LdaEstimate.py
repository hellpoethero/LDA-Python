import random
from variationalLDA.LdaModel import LdaModel
from variationalLDA.LdaInference import LdaInference
from variationalLDA.LdaAlpha import LdaAlpha
import numpy as np


class LdaEstimate:
	LAG = 10
	NUM_INIT = 1
	EM_CONVERGED = 0
	EM_MAX_ITER = 0
	ESTIMATE_ALPHA = 0
	INITIAL_ALPHA = 0
	K = 0

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

	@staticmethod
	def doc_em(doc, gamma, model, next_model):
		phi = [[0.0] * model.num_topics] * doc.length
		likelihood = LdaInference.inference(doc, model, gamma, phi)
		for n in range(0, doc.length):
			for k in range(0, model.num_topics):
				next_model.class_word[k][doc.words[n]] += doc.counts[n] * phi[n][k]
				next_model.class_total[k] += doc.counts[n] * phi[n][k]
		return likelihood

	@staticmethod
	def save_gamma(filename, gamma, num_docs, num_topics):
		pass

	@staticmethod
	def run_em(start, directory, corpus):
		likelihood, likelihood_old = np.NINF
		converged = 1
		var_gamma = [[0] * LdaEstimate.K] * corpus.num_docs
		model = LdaEstimate.initial_model(start, corpus, LdaEstimate.K, LdaEstimate.INITIAL_ALPHA)

		i = 0
		while converged > LdaEstimate.EM_CONVERGED or (i <= 2 and i <= LdaEstimate.EM_MAX_ITER):
			likelihood = 0
			next_model = LdaModel()
			next_model.set_model(model.num_term, model.num_topics)
			next_model.alpha = LdaEstimate.INITIAL_ALPHA
			for d in range(0, corpus.num_docs):
				if d % 100 == 0:
					likelihood += LdaEstimate.doc_em(corpus.docs[d], var_gamma[d], model, next_model)
			if LdaEstimate.ESTIMATE_ALPHA == 1:
				LdaAlpha.maximize_alpha(var_gamma, next_model, corpus.num_docs)

			model = next_model
			converged = (likelihood_old - likelihood) / likelihood
			likelihood_old = likelihood

			if i % LdaEstimate.LAG == 0:
				pass

			return model

	@staticmethod
	def read_settings(filename):
		pass

	@staticmethod
	def infer(model_root, save, corpus):
		model = LdaModel()
		model.load_model(model_root)
		var_gamma = [[0.0] * model.num_topics] * corpus.num_docs

		for d in range(0, corpus.num_docs):
			doc = corpus.docs[d]
			phi = [[0.0] * doc.length] * model.num_topics
			likelihood = LdaInference.inference(doc, model, var_gamma, phi)
