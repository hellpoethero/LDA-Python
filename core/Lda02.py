import datetime
import pandas as pd
import numpy as np
import random
from scipy import sparse
from scipy.special import digamma
from scipy.sparse import csr_matrix
from time import time
from functools import reduce


def init(df, Ks, iters, top_k, datasetName, outFile, folder_name):
	num_record = len(df)
	print(num_record)

	nodelist = list(set(df.user_id.tolist()))
	venuelist = list(set(df.loc_id.tolist()))

	venueSize = len(venuelist)

	print(venueSize)
	print(len(nodelist))

	print("Calculate the history")
	history = [[0]] * num_record
	start = time()
	temp_his = np.zeros(venueSize)
	temp_count = 0
	repeated = np.zeros(num_record)
	prev_user = -1
	history_index = 0

	for r in zip(df['user_id'], df['train'], df['loc_id']):
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
	df['history'] = history
	df['repeated'] = repeated

	# print(df_table[df_table['user_id'] == 0])

	print(time() - start)
	print("done")

	run(df, 2, 1, top_k, datasetName, outFile, folder_name)
	for K in Ks:
		for iteration in iters:
			run(df, K, iteration, top_k, datasetName, outFile, folder_name)


def run(df, K, iteration, top_k, datasetName, outFile, folder_name):
	num_record = len(df)
	print(num_record)

	df_inference = df[['user_id', 'loc_id', 'train', 'history', 'repeated']]
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
		[['user_id', 'loc_id', 'phi', 'train', 'gamma', 'history', 'repeated']]

	df_table_training = df_table[df_table.train == 1]

	# print(len(set(df_table[df_table.train == 1]['loc_id'].tolist())))
	df_table_testing = df_table[df_table.train == 0]
	df_table_backup = df_table
	df_table = df_table_training

	def toSparseMatrix(x):
		I = range(K)
		J = np.full(K, x.loc_id)
		V = x.phi
		return sparse.coo_matrix((V, (I, J)), shape=(K, venueSize))

	#
	# def updateRoleVenue(x):
	#     roleVenue = np.ones([K, len(venuelist)]) / len(venuelist)
	#     for index, row in x.iterrows():
	#         if row['train'] > 0:
	#             roleVenue[:, row['loc_id']] += row['phi']
	#     roleVenue /= roleVenue.sum(axis=1)[:, np.newaxis]
	#     return roleVenue

	def updateRoleVenue(x):
		roleVenue = np.ones([K, len(venuelist)]) / len(venuelist) / (K - 1)

		a = x.groupby('loc_id')['phi'].apply(lambda x: x.sum()).sort_index()
		b = np.array(a.tolist())
		b = b.T
		#	roleVenue[:,:-(roleVenue.shape[1] - b.shape[1])] += b
		roleVenue += b
		#	roleVenue = b
		roleVenue /= roleVenue.sum(axis=1)[:, np.newaxis]

		#	roleVenue = 0.8* np.array([g, ] * K) + 0.2 * roleVenue
		roleVenue[K - 1] = g
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

	def multiplyArray_backup(x):
		return tuple(np.multiply(x['temp'], x['gamma']))

	def multiplyArray(x):  ## x[1] gamma
		temp = digamma(x[1])
		temp = np.exp(temp)
		a = np.multiply(x[0], temp)
		return a / (a.sum())

	input_temp = df_table[['temp', 'gamma']].values
	df_table['phi'] = np.apply_along_axis(multiplyArray, 1, input_temp).tolist()
	df_table['phi'] = df_table['phi'].apply(lambda x: np.array(x))

	# roleVenue = updateRoleVenue(df_table)

	vf_test = np.vectorize(f_test)

	### update global
	g = df_table.groupby(['loc_id'])['user_id'].apply(np.unique).apply(lambda x: len(x)).sort_index()
	# g = df_table.groupby('loc_id')['phi'].apply(lambda x: x.count()).sort_index()
	g = np.array(g.tolist())
	g = g.T
	g = g / float(sum(g))

	print("optimize")

	start = time()
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

	print("Optimize done")
	print(time()-start)
	valid_user = df_table_backup.groupby('user_id')['train'].mean()
	valid_user = valid_user[(valid_user > 0.001) & (valid_user < 0.999)]
	valid_user = list(set(valid_user.index))

	top_k = 100

	df_inference_testing = df_table_testing[['user_id', 'loc_id', 'phi', 'train', 'history', 'repeated']]
	df_table_testing = pd.merge(df_inference_testing, dfgamma, how='left', left_on=['user_id'], right_on=['user_id']) \
		[['user_id', 'loc_id', 'phi', 'gamma', 'train', 'history', 'repeated']]

	df_table_testing = df_table_testing[df_table_testing['user_id'].isin(valid_user)].reset_index()
	in_top = np.zeros(len(df_table_testing))
	ranking_avg = np.zeros(len(df_table_testing))
	print("calculate ranking")

	def rates_to_exp_order(rates, argsort, exp_order, M):
		prev_score = 0
		prev_idx = 0
		prev_val = rates[argsort[0]]
		for i in range(1, M):
			if prev_val == rates[argsort[i]]:
				continue

			tmp = 0
			for j in range(prev_idx, i):
				exp_order[argsort[j]] = prev_score + 1
				tmp += 1

			prev_score += tmp
			prev_val = rates[argsort[i]]
			prev_idx = i

		# For the last equalities
		for j in range(prev_idx, i + 1):
			exp_order[argsort[j]] = prev_score + 1

	avg_gamma = dfgamma.mean().gamma

	for index, row in df_table_testing.iterrows():
		#	if np.isnan(np.min(row['gamma']))  :
		#		print("here")
		#	else:
		roleVenue2 = np.array(roleVenue)
		if np.isnan(np.min(row['gamma'])):
			roleVenue2[0] = np.ones(venueSize) / venueSize
			array = np.dot(avg_gamma / sum(avg_gamma), roleVenue2)
		else:
			roleVenue2[0] = row['history'].toarray()
			array = np.dot(row['gamma'] / sum(row['gamma']), roleVenue2)
		scores = array
		argsort = np.argsort(-scores)
		exp_order = np.zeros(scores.shape)
		rates_to_exp_order(scores, argsort, exp_order, len(scores))
		# order = array.argsort()
		ranks = exp_order
		# ranks = venueSize - order.argsort()
		if ranks[row['loc_id']] <= top_k:
			in_top[index] = 1
		ranking_avg[index] = ranks[row['loc_id']]

	print("K=" + str(K))
	df_table_testing['in_top'] = in_top
	df_table_testing['rank'] = ranking_avg

	avg_rank = reduce(lambda x, y: x + y, ranking_avg) / len(ranking_avg)
	print(avg_rank)

	recall = np.mean(df_table_testing.groupby('user_id')['in_top'].mean().values)
	print(recall)
	recall_all = sum(df_table_testing['in_top']) / len(df_table_testing)
	print(recall_all)

	recall_repeated = np.mean(df_table_testing[df_table_testing['repeated'] == 1].groupby('user_id')['in_top'].mean().values)
	print(recall_repeated)
	recall_new = np.mean(df_table_testing[df_table_testing['repeated'] == 0].groupby('user_id')['in_top'].mean().values)
	print(recall_new)

	outFile.write(str(K) + "," + str(iteration) + "," + str(recall) + "," + str(recall_all) + "," + str(recall_repeated)
				  + "," + str(recall_new) + "," + str(int(avg_rank)) + ","
				  + str(np.mean(df_table_testing[df_table_testing['repeated'] == 1]['rank'])) + ","
				  + str(np.mean(df_table_testing[df_table_testing['repeated'] == 0]['rank'])) + "\n")

	df_table_testing[['user_id', 'loc_id', 'repeated', 'rank']].to_csv(
		folder_name + "/" + datasetName[:-1] + "_K" + str(K) + "_I" + str(iteration) + ".csv")


