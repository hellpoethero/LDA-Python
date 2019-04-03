import variationalLDA.LdaModel as LdaModel
import variationalLDA.Utils as Utils
import numpy as np


class LdaAlpha:
	@staticmethod
	def objective(s, suff_stats, num_docs, num_topics):
		r = ((s / num_topics) - 1) * suff_stats / num_docs
		r += Utils.lgamma(s) - num_topics * Utils.lgamma(s / num_topics)
		return r

	@staticmethod
	def gradient(s, suff_stats, num_docs, num_topics):
		r = num_docs * (Utils.digamma(s) - Utils.digamma(s / num_topics)) + suff_stats / num_topics;
		return r

	@staticmethod
	def gradient_ascent(s, suff_stats, num_docs, num_topics):
		step_size = 0.1
		s = 1
		f = LdaAlpha.objective(s, suff_stats, num_docs, num_topics)
		old_f = f - 1
		while (f - old_f) / np.abs(old_f) > 0.0001:
			old_s = s
			old_f = f
			s = s + step_size * LdaAlpha.gradient(s, suff_stats, num_docs, num_topics)
			f = LdaAlpha.objective(s, suff_stats, num_docs, num_topics)
			while s < 0 or f < old_f:
				s = old_s
				step_size = step_size / 2
				s = s + step_size * LdaAlpha.gradient(s, suff_stats, num_docs, num_topics)
				f = LdaAlpha.objective(s, suff_stats, num_docs, num_topics)
		return s

	@staticmethod
	def maximize_alpha(gamma, model: LdaModel, num_docs):
		suff_stats = 0
		for d in range(0, num_docs):
			gamma_sum = 0
			for k in range(0, model.num_topics):
				gamma_sum += gamma[d][k]
				suff_stats += Utils.digamma(gamma[d][k])
			suff_stats -= model.num_topics * Utils.digamma(gamma_sum)
		s = LdaAlpha.gradient_ascent(model.alpha * model.num_topics, suff_stats, num_docs, model.num_topics)
		model.alpha = s / model.num_topics


