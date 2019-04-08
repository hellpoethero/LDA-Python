import SortedWord


class LdaModel:

	def __init__(self):
		self.alpha = 0.0
		self.class_word = []  # word_topic distribution
		self.class_total = []
		self.num_topics = 0
		self.num_term = 0
		self.vocab = []

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

	def read_vocab(self, filename):
		with open(filename, "r") as inputFile:
			for line in inputFile:
				self.vocab.append(line.rstrip())

	def load_model(self, model_root):
		pass

	def save_model(self, model_root):
		pass

	def print_word_topic_distribution(self):
		words = []
		for n in range(0, self.num_term):
			sorted_word = SortedWord.SortedWord(n, self.vocab[n], self.class_word[0][n])
			words.append(sorted_word)
		print(len(words))
		sorted_words = sorted(words, key=lambda x: x.count, reverse=True)
		for n in range(0, 10):
			print(sorted_words[n])
