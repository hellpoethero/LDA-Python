class LdaModel:
	alpha = 0.0
	class_word = []
	class_total = []
	num_topics = 0
	num_term = 0

	def __init__(self, numTerms, numTopics):
		pass

	def set_model(self, num_term, num_torpic):
		self.num_term = num_term
		self.num_topics = num_torpic
		self.alpha = 1.0

		self.class_total = [0] * self.num_topics
		self.class_word = [[0] * self.num_term] * self.num_topics

	def load_model(self, model_root):
		pass

	def save_model(self, model_root):
		pass
