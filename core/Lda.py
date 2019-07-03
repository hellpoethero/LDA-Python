import datetime
import pandas as pd
import numpy as np
import random
from scipy import sparse
from scipy.special import digamma
from scipy.sparse import csr_matrix
from time import time
from functools import reduce


def run(df, K, iteration, top_k, datasetName, outFile):
	num_record = len(df)
	print(num_record)

	df_inference = df[['user_id', 'loc_id', 'train']]
	temp = np.random.dirichlet(np.ones(K) * 1000, len(df)).tolist()
	temparray = [np.array(row) for row in temp]
	df_inference['phi'] = temparray

	nodelist = list(set(df_inference.user_id.tolist()))
	venuelist = list(set(df_inference.loc_id.tolist()))

	venueSize = len(venuelist)

	print(venueSize)
	print(len(nodelist))

	roleVenue = np.random.dirichlet(np.ones(venueSize), K)
	# roleVenue = np.ones([K, venueSize]) / venueSize

	tempp = (np.random.dirichlet(np.ones(K) * 1000, len(nodelist)) * K).tolist()
	tempparray = [np.array(row) for row in tempp]
	seriesgamma = tempparray
	dfgamma = pd.DataFrame(list(zip(nodelist, seriesgamma)), columns=['user_id', 'gamma'])

	df_table = pd.merge(df_inference, dfgamma, how='left', left_on=['user_id'], right_on=['user_id']) \
		[['user_id', 'loc_id', 'phi', 'train', 'gamma']]

	print("Calculate the history")
	history = [[0]] * num_record
	start = time()
	temp_his = np.zeros(venueSize)
	temp_count = 0
	repeated = np.zeros(num_record)
	prev_user = -1
	history_index = 0

	for r in zip(df_table['user_id'], df_table['train'], df_table['loc_id']):
		if history_index % 1000 == 0:
			print(history_index)
		if prev_user == -1 or r[0] != prev_user:
			temp_his = csr_matrix((1, venueSize), dtype=int)
			temp_count = 0
			history[history_index] = csr_matrix(temp_his)
		else:
			if temp_count == 0:  # added for divide by zero
				history[history_index] = temp_his
			else:
				history[history_index] = temp_his / temp_count
		if r[1] == 1:
			temp_his[(0, r[2])] += 1
			temp_count += 1
		else:
			if temp_his[(0, r[2])] > 0:
				repeated[history_index] = 1
		prev_user = r[0]
		history_index += 1
	df_table['history'] = history
	df_table['repeated'] = repeated

	# print(df_table[df_table['user_id'] == 0])

	print(time() - start)
	print("done")
	# print(df_table['history'])

	df_table_training = df_table[df_table.train == 1]

	# print(len(set(df_table[df_table.train == 1]['loc_id'].tolist())))
	df_table_testing = df_table[df_table.train == 0]
	df_table_backup = df_table
	df_table = df_table_training

	def updateRoleVenue(x):
		roleVenue = np.ones([K, len(venuelist)]) / len(venuelist)

		a = x.groupby('loc_id')['phi'].apply(lambda x: x.sum()).sort_index()
		b = np.array(a.tolist())
		b = b.T
		roleVenue += b
		#	roleVenue = b
		roleVenue /= roleVenue.sum(axis=1)[:, np.newaxis]

		#	roleVenue = 0.01 * np.array([g, ] * K) + 0.99 * roleVenue
		return roleVenue

	df_table['temp'] = df_table['loc_id'].apply(lambda x: roleVenue[:, x])

	def f_test(x):
		temp_history = x[2]
		temp_loc = x[1]
		temp = x[0]
		temp[0] = temp_history[(0, temp_loc)]
		return temp

	input_temp = df_table[['temp', 'loc_id', 'history']].values
	df_table['temp'] = np.apply_along_axis(f_test, 1, input_temp).tolist()
	df_table['temp'] = df_table['temp'].apply(lambda x: np.array(x))

	def multiplyArray(x):  ## x[1] gamma
		temp = digamma(x[1])
		temp = np.exp(temp)
		a = np.multiply(x[0], temp)
		return a / (a.sum())

	input_temp = df_table[['temp', 'gamma']].values
	df_table['phi'] = np.apply_along_axis(multiplyArray, 1, input_temp).tolist()
	df_table['phi'] = df_table['phi'].apply(lambda x: np.array(x))

	g = df_table.groupby('loc_id')['phi'].apply(lambda x: x.count()).sort_index()
	g = np.array(g.tolist())
	g = g.T
	g = g / float(sum(g))

	print("optimize")

	for i in range(iteration):
		print("iter " + str(i + 1))

		dfgamma = df_table.groupby(['user_id'])['phi'].apply(
			lambda x: x.sum() + np.ones(K) / K).to_frame().reset_index()
		dfgamma.columns = ['user_id', 'gamma']

		df_table['temp'] = df_table['loc_id'].apply(lambda x: roleVenue[:, x])

		input_temp = df_table[['temp', 'loc_id', 'history']].values
		df_table['temp'] = np.apply_along_axis(f_test, 1, input_temp).tolist()
		df_table['temp'] = df_table['temp'].apply(lambda x: np.array(x))

		input_temp = df_table[['temp', 'gamma']].values
		df_table['phi'] = np.apply_along_axis(multiplyArray, 1, input_temp).tolist()
		df_table['phi'] = df_table['phi'].apply(lambda x: np.array(x))

		df_inference = df_table[['user_id', 'loc_id', 'phi', 'train', 'history']]

		df_table = pd.merge(df_inference, dfgamma, how='left', left_on=['user_id'], right_on=['user_id']) \
			[['user_id', 'loc_id', 'phi', 'gamma', 'train', 'history']]

		roleVenue = updateRoleVenue(df_table)

	print(len(roleVenue[0]))
	print("Optimize done")
	valid_user = df_table_backup.groupby('user_id')['train'].mean()
	valid_user = valid_user[(valid_user > 0.001) & (valid_user < 0.999)]
	valid_user = list(set(valid_user.index))

	df_inference_testing = df_table_testing[['user_id', 'loc_id', 'phi', 'train', 'history', 'repeated']]
	df_table_testing = pd.merge(df_inference_testing, dfgamma, how='left', left_on=['user_id'], right_on=['user_id']) \
		[['user_id', 'loc_id', 'phi', 'gamma', 'train', 'history', 'repeated']]

	df_table_testing = df_table_testing[df_table_testing['user_id'].isin(valid_user)].reset_index()
	in_top = np.zeros(len(df_table_testing))
	ranking_avg = np.zeros(len(df_table_testing))
	print("calculate ranking")

	print(len(roleVenue[0]))

	for index, row in df_table_testing.iterrows():
		roleVenue2 = np.array(roleVenue)
		roleVenue2[0] = row['history'].toarray()
		array = np.dot(row['gamma'] / sum(row['gamma']), roleVenue2)
		order = array.argsort()
		ranks = venueSize - order.argsort()
		if ranks[row['loc_id']] <= top_k:
			in_top[index] = 1
		ranking_avg[index] = ranks[row['loc_id']]

	avg_rank = reduce(lambda x, y: x + y, ranking_avg) / len(ranking_avg)
	print(avg_rank)
	print(K)
	df_table_testing['in_top'] = in_top
	df_table_testing['rank'] = ranking_avg

	recall = np.mean(df_table_testing.groupby('user_id')['in_top'].mean().values)
	print(recall)
	recall_all = sum(df_table_testing['in_top']) / len(df_table_testing)
	print(recall_all)

	recall_repeated = np.mean(df_table_testing[df_table_testing['repeated'] == 1].groupby('user_id')['in_top'].mean().values)
	print(recall_repeated)
	recall_new = np.mean(df_table_testing[df_table_testing['repeated'] == 0].groupby('user_id')['in_top'].mean().values)
	print(recall_new)

	# outFile.write(str() + "," + str() + "," + str() + "," + str() + "," + "\n")

	df_table_testing[['user_id', 'loc_id', 'rank']].to_csv(
		"D:/Research/Project/LDA/results/" + datasetName[:-1] + "_K" + str(K) + "_I" + str(iteration) + ".csv")


