import datetime
import pandas as pd
import numpy as np
import random
from scipy import sparse
from scipy.special import digamma
from scipy.sparse import csr_matrix
from time import time


def read(path, dataset):
	trainFile = path + dataset + "train.csv"
	validationFile = path + dataset + "validation.csv"
	testFile = path + dataset + "test.csv"
	train = pd.read_csv(trainFile, header=None)
	validation = pd.read_csv(validationFile, header=None)
	test = pd.read_csv(testFile, header=None)

	train.columns = ["user_id", "loc_id", "count"]
	validation.columns = ["user_id", "loc_id", "count"]
	test.columns = ["user_id", "loc_id", "count"]

	train_df = pd.concat([function01(train, 1),
						  function01(validation, 1)])
	# test_df = function01(test, 1).sample(frac=1, random_state=123).sort_values(by=['user_id'], ascending=[1])
	test_df = function01(test, 0)
	test_df = test_df[test_df['loc_id'].isin(train_df['loc_id'].tolist())]

	# change = test_df.sample(int(test_df.shape[0] * 0.1), random_state=123).index
	# test_df.loc[change, 'train'] = 0

	# prev_user = -1
	# index = 0
	# train_new = np.array(test_df['train'].values)
	# for r in zip(test_df['user_id']):
	# 	if prev_user != -1 and r[0] != prev_user:
	# 		train_new[index-1] = 0
	# 	prev_user = r[0]
	# 	index += 1
	# train_new[len(train_new)-1] = 0
	# test_df['train'] = train_new
	# print(test_df[['user_id', 'train']])

	df = pd.concat([train_df, test_df])

	loc_df = pd.DataFrame(df['loc_id'].unique())
	loc_df['index1'] = loc_df.index
	loc_df.columns = ['loc_id', 'index1']

	df = pd.merge(df, loc_df, how='inner', left_on=['loc_id'], right_on=['loc_id']) \
		[['user_id', 'index1', 'train']]
	df.columns = ['user_id', 'loc_id', 'train']
	df = df.sort_values(by=['user_id', 'train'], ascending=[1, 0])
	return df


def function01(data, train):
	array = []
	for index, row in data.iterrows():
		temp = [[row['user_id'], row['loc_id'], train]] * row['count']
		array.extend(temp)

	df = pd.DataFrame(array)
	# df = df.sample(frac=0.4, replace=True, random_state=1)
	df.columns = ["user_id", "loc_id", "train"]
	return df