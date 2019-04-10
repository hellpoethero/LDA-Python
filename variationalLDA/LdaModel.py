import numpy as np
import time
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

		self.class_total = np.zeros([self.num_topics])
		self.class_word = np.zeros([self.num_topics, self.num_term])

	def read_vocab(self, filename):
		with open(filename, "r") as inputFile:
			for line in inputFile:
				self.vocab.append(line.rstrip())

	def load_model(self, model_root):
		pass

	def save_model(self, model_root):
		pass

	def print_word_topic_distribution(self):
		start = time.time()
		words = np.empty([self.num_term], dtype=SortedWord.SortedWord)
		for n in range(0, self.num_term):
			sorted_word = SortedWord.SortedWord(n, self.vocab[n], 0)
			words[n] = sorted_word
		for k in range(0, self.num_topics):
			for n in range(0, self.num_term):
				words[n].count = self.class_word[k][n]
			sorted_words = sorted(words, key=lambda x: x.count, reverse=True)
			for n in range(0, 20):
				print(sorted_words[n])
			print("----------")
		end = time.time()
		print(end - start)
