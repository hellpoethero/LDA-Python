import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
from gensim import corpora, models
from pprint import pprint

data = pd.read_csv('D:/Research/Dataset/abcnews-date-text.csv',
                   error_bad_lines=False)
data_text = data[['headline_text']]
data_text['index'] = data_text.index
documents = data_text

print(len(documents))
# print(documents[:5])

np.random.seed(2018)

stemmer = PorterStemmer()


def lemmatize_stemming(text):
  return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
  result = []
  for token in gensim.utils.simple_preprocess(text):
    if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
      result.append(lemmatize_stemming(token))
  return result


doc_sample = documents[documents['index'] == 4310].values[0][0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
  words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

processed_docs = documents['headline_text'].map(preprocess)
print(processed_docs[:10])

dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

count = 0
for k, v in dictionary.iteritems():
  print(k, v)
  count += 1
  if count > 10:
    break

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print(bow_corpus[4310])
bow_doc_4310 = bow_corpus[4310]
for i in range(len(bow_doc_4310)):
  print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0],
                                                   dictionary[
                                                     bow_doc_4310[i][0]],
                                                   bow_doc_4310[i][1]))

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

# for doc in corpus_tfidf:
#     pprint(doc)
#     break

print("run lda")

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10,
                                       id2word=dictionary, passes=2, workers=2)

# for idx, topic in lda_model.print_topics(-1):
#     print('Topic: {} \nWords: {}'.format(idx, topic))
