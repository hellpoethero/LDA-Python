import numpy as np
from LdaModel import LdaModel
import Utils


class LdaInference:
	VAR_CONVERGED = 0
	VAR_MAX_ITER = 0

	@staticmethod
	def inference(doc, model: LdaModel, var_gamma, phi):
		converged = 1
		likelihood = 0
		likelihood_old = -np.NINF
		old_phi = np.zeros([model.num_topics])

		for k in range(0, model.num_topics):
			var_gamma[k] = model.alpha + doc.total / float(model.num_topics)
			for n in range(0, doc.length):
				phi[n][k] = 1.0 / model.num_topics

		var_iter = 0
		while (converged > LdaInference.VAR_CONVERGED) and (var_iter < LdaInference.VAR_MAX_ITER):
			var_iter += 1
			for n in range(0, doc.length):
				phi_sum = 0
				for k in range(0, model.num_topics):
					old_phi[k] = phi[n][k]

					if model.class_word[k][doc.words[n]] > 0:
						phi[n][k] = Utils.digamma(var_gamma[k]) \
									+ np.log(model.class_word[k][doc.words[n]]) \
									- np.log(model.class_total[k])
					else:
						phi[n][k] = Utils.digamma(var_gamma[k]) - 100

					if k > 0:
						phi_sum = Utils.log_sum(phi_sum, phi[n][k])
					else:
						phi_sum = phi[n][k]

				for k in range(0, model.num_topics):
					phi[n][k] = np.exp(phi[n][k] - phi_sum)
					var_gamma[k] = var_gamma[k] + doc.counts[n] * (phi[n][k] - old_phi[k])

			likelihood = LdaInference.compute_likelihood(doc, model, phi, var_gamma)

			converged = (likelihood_old - likelihood) / likelihood
			likelihood_old = likelihood
		return likelihood

	@staticmethod
	def compute_likelihood(doc, model, phi, var_gamma):
		likelihood = 0
		dig_sum = 0
		var_gamma_sum = 0
		dig = np.zeros([model.num_topics])

		for k in range(0, model.num_topics):
			dig[k] = Utils.digamma(var_gamma[k])
			var_gamma_sum += var_gamma[k]

		dig_sum = Utils.digamma(var_gamma_sum)

		likelihood = Utils.lgamma(model.alpha * model.num_topics) \
					 - model.num_topics * Utils.lgamma(model.alpha) \
					 - Utils.lgamma(var_gamma_sum)

		for k in range(0, model.num_topics):
			likelihood += \
				(model.alpha - 1) * (dig[k] - dig_sum) \
				+ Utils.lgamma(var_gamma[k]) \
				- (var_gamma[k] - 1) * (dig[k] - dig_sum)

			for n in range(0, doc.length):
				if model.class_word[k][doc.words[n]] > 0:
					if phi[n][k] > 0:
						likelihood += \
							doc.counts[n] * (phi[n][k] * ((dig[k] - dig_sum) - np.log(phi[n][k]))) \
							+ np.log(model.class_word[k][doc.words[n]]) \
							- np.log(model.class_total[k])
		return likelihood
