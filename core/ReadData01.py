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

	train_df = function01(train, 1)
	validation_df = function01(validation, 1)
	test_df = function01(test, 0)
	train_user_set = set(train_df['user_id'].tolist())
	validation_user_set = set(validation_df['user_id'].tolist())
	user_of_interest = train_user_set.intersection(validation_user_set)

	train_df = train_df[train_df['user_id'].isin(user_of_interest)]
	validation_df = validation_df[validation_df['user_id'].isin(user_of_interest)]
	test_df = test_df[test_df['user_id'].isin(user_of_interest)]

	train_df = pd.concat([train_df, validation_df])
	test_df = test_df[test_df['loc_id'].isin(train_df['loc_id'].tolist())]

	df = pd.concat([train_df, test_df])

	loc_df = pd.DataFrame(df['loc_id'].unique())
	loc_df['index1'] = loc_df.index
	loc_df.columns = ['loc_id', 'index1']

	df = pd.merge(df, loc_df, how='inner', left_on=['loc_id'], right_on=['loc_id']) \
		[['user_id', 'index1', 'train']]
	df.columns = ['user_id', 'loc_id', 'train']
	df = df.sample(frac=1, random_state=123).sort_values(by=['user_id', 'train'], ascending=[1, 0])

	return df



def read01(path, dataset):
	trainFile = path + dataset + "train.csv"
	validationFile = path + dataset + "validation.csv"
	testFile = path + dataset + "test.csv"
	train = pd.read_csv(trainFile, header=None)
	validation = pd.read_csv(validationFile, header=None)
	test = pd.read_csv(testFile, header=None)

	train.columns = ["user_id", "loc_id", "count"]
	validation.columns = ["user_id", "loc_id", "count"]
	test.columns = ["user_id", "loc_id", "count"]

	train_valid = pd.concat([train, validation])
	d = train_valid.groupby(['loc_id'])['user_id'].apply(np.unique).apply(lambda x: len(x)).reset_index()
	loc_id_list = d[d['user_id'] >= 50]['loc_id'].to_list()

	train_user_set = set(train_valid['user_id'].tolist())
	test_user_set = set(test['user_id'].tolist())
	user_of_interest = train_user_set.intersection(test_user_set)
	train_valid = train_valid[train_valid['user_id'].isin(user_of_interest) & train_valid['loc_id'].isin(loc_id_list)]

	d = train_valid.groupby(['loc_id'])['user_id'].apply(np.unique).apply(lambda x: len(x)).reset_index()
	loc_id_list = d[d['user_id'] >= 50]['loc_id'].to_list()

	train_user_set = set(train_valid['user_id'].tolist())
	test_user_set = set(test['user_id'].tolist())
	user_of_interest = train_user_set.intersection(test_user_set)
	train_valid = train_valid[train_valid['user_id'].isin(user_of_interest) & train_valid['loc_id'].isin(loc_id_list)]

	test = test[test['user_id'].isin(train_valid['user_id'].to_list())]
	test = test[test['loc_id'].isin(train_valid['loc_id'].to_list())]

	df = pd.concat([function01(train_valid, 1), function01(test, 0)])
	loc_df = pd.DataFrame(df['loc_id'].unique())
	loc_df['index1'] = loc_df.index
	loc_df.columns = ['loc_id', 'index1']

	user_df = pd.DataFrame(df['user_id'].unique())
	user_df['index2'] = user_df.index
	user_df.columns = ['user_id', 'index2']
	new_df = pd.merge(df, loc_df, how='inner', left_on=['loc_id'], right_on=['loc_id']) \
		[['user_id', 'index1', 'train']]
	new_df = pd.merge(new_df, user_df, how='inner', left_on=['user_id'], right_on=['user_id']) \
		[['index2', 'index1', 'train']]

	new_df.columns = ['user_id', 'loc_id', 'train']
	new_df = new_df.sample(frac=1, random_state=123).sort_values(by=['user_id', 'train'], ascending=[1, 0]).reset_index()

	return new_df

def function01(data, train):
	array = []
	for index, row in data.iterrows():
		temp = [[row['user_id'], row['loc_id'], train]] * row['count']
		array.extend(temp)

	df = pd.DataFrame(array)
	# df = df.sample(frac=0.4, replace=True, random_state=1)
	df.columns = ["user_id", "loc_id", "train"]
	return df