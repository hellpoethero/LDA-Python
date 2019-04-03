from variationalLDA.Document import Document


class Corpus:
	docs = []
	num_terms = 0
	num_docs = 0

	def read(self, filename):
		with open(filename, "r") as inputFile:
			for line in inputFile:
				fields = line.rstrip().split(" ")
				doc_len = int(fields[0])
				self.num_docs += 1
				doc = Document(doc_len)
				for field in fields[1:]:
					word = int(field.split(":")[0])
					count = int(field.split(":")[1])

					doc.total += count
					doc.words.append(word)
					doc.counts.append(count)

					if word >= self.num_terms:
						self.num_terms = word + 1
				self.docs.append(doc)
