class LdaModel:

	def __init__(self):
		self.alpha = 0.0
		self.class_word = []
		self.class_total = []
		self.num_topics = 0
		self.num_term = 0

	def set_model(self, num_term, num_topic):
		self.num_term = num_term
		self.num_topics = num_topic
		self.alpha = 1.0

		for k in range(0, self.num_topics):
			self.class_total.append(0)
			temp_array = []
			for n in range(0, self.num_term):
				temp_array.append(0)
			self.class_word.append(temp_array)

	def load_model(self, model_root):
		pass

	def save_model(self, model_root):
		pass
