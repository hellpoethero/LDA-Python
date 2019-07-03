import pandas as pd
import numpy as np
import datetime
from scipy.special import digamma
import itertools
import random
from scipy import sparse
from functools import reduce
import time

K = 50
file_path = "C:/Users/ducnm/Downloads/SFcheckin_0509_2.csv"
df = pd.read_csv(file_path, sep=',', index_col=0)
df_inference = df[['user_id', 'loc_id']]
temp = np.random.dirichlet(np.ones(K) * 10, len(df)).tolist()
# temp = (np.ones((len(df),K))/K).tolist()
temparray = [np.array(row) for row in temp]
df_inference['phi'] = temparray

nodelist = list(set(df_inference.user_id.tolist()))
venuelist = list(set(df_inference.loc_id.tolist()))

venueSize = len(venuelist)

roleVenue = np.ones([K, len(venuelist)]) / len(venuelist)

tempp = (np.random.dirichlet(np.ones(K) * 100, len(nodelist)) * K).tolist()
tempparray = [np.array(row) for row in tempp]
seriesgamma = tempparray
dfgamma = pd.DataFrame(list(zip(nodelist, seriesgamma)), columns=['user_id', 'gamma'])

df_table = pd.merge(df_inference, dfgamma, how='left', left_on=['user_id'], right_on=['user_id']) \
	[['user_id', 'loc_id', 'phi', 'gamma']]

df_table['train'] = 1
change = df_table.sample(int(df_table.shape[0] * 0.1), random_state=123).index
df_table.loc[change, 'train'] = 0


### backup for original model

def toSparseMatrix(x):
	I = range(K)
	J = np.full(K, x.loc_id)
	V = x.phi
	return sparse.coo_matrix((V, (I, J)), shape=(K, venueSize))


def updateRoleVenue(x):
	roleVenue = np.ones([K, len(venuelist)]) / len(venuelist)
	for index, row in x.iterrows():
		if (row['train'] > 0):
			roleVenue[:, row['loc_id']] += row['phi']
	roleVenue /= roleVenue.sum(axis=1)[:, np.newaxis]
	return roleVenue


df_table['temp'] = df_table['loc_id'].apply(lambda x: roleVenue[:, x])

def multiplyArray(x):
	a = np.multiply(x[0], x[1])
	return a / (a.sum())

input_temp = df_table[['temp', 'gamma']].values
df_table['phi'] = np.apply_along_axis(multiplyArray, 1, input_temp).tolist()
df_table['phi'] = df_table['phi'].apply(lambda x: np.array(x))

roleVenue = updateRoleVenue(df_table)

for i in range(1):
	print("iter " + str(i + 1))

	dfgamma = df_table.groupby(['user_id'])['phi'].apply(lambda x: x.sum() + np.ones(K) / K).to_frame().reset_index()
	dfgamma.columns = ['user_id', 'gamma']

	df_table['temp'] = df_table['loc_id'].apply(lambda x: roleVenue[:, x])

	start = time.time()
	# df_table['phi'] = df_table.apply(multiplyArray, axis=1) \
	#     .apply(np.array).apply(lambda x: x / x.sum())
	input_temp = df_table[['temp', 'gamma']].values
	df_table['phi'] = np.apply_along_axis(multiplyArray, 1, input_temp).tolist()
	df_table['phi'] = df_table['phi'].apply(lambda x: np.array(x))
	print(time.time() - start)

	df_inference = df_table[['user_id', 'loc_id', 'phi', 'train']]
	df_table = pd.merge(df_inference, dfgamma, how='left', left_on=['user_id'], right_on=['user_id']) \
		[['user_id', 'loc_id', 'phi', 'gamma', 'train']]
	roleVenue = updateRoleVenue(df_table)

ranking_avg = []

true_prediction_count = 0
prediction_count = 0
top_k = 100
for index, row in df_table.iterrows():
	if row['train'] < 1:
		array = np.dot(row['gamma'] / sum(row['gamma']), roleVenue)
		order = array.argsort()
		ranks = venueSize - order.argsort()
		if ranks[row['loc_id']] <= top_k:
			true_prediction_count += 1
		ranking_avg.append(ranks[row['loc_id']])
		prediction_count += 1

print(reduce(lambda x, y: x + y, ranking_avg) / len(ranking_avg))
print(K)
print(float(true_prediction_count) / prediction_count)
