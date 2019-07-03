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
# file_path = "C:/Users/ducnm/Downloads/SFcheckin_0604_3.csv"
file_path = "C:/Users/ducnm/Downloads/NYcheckin_0606.csv"
df = pd.read_csv(file_path, sep=',', index_col=0)

df['train'] = 1
# change = df.sample(int(df.shape[0]*0.1), random_state=123).index
# df.loc[change, 'train'] = 0
df.loc[df.time.str.startswith('2010-09'), 'train'] = 0
df.loc[df.time.str.startswith('2010-10'), 'train'] = 0

df_inference = df[['user_id', 'loc_id', 'train']]
temp = np.random.dirichlet(np.ones(K) * 10, len(df)).tolist()
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
	[['user_id', 'loc_id', 'phi', 'train', 'gamma']]

df_table['history'] = [[0]] * df_table.shape[0]
df_table['history'] = df_table['history'].apply(lambda x: np.array(x))

history = [[0]] * df_table.shape[0]
loc_temp = []
loc_count_temp = []
for i in range(0, len(df_table)):
	row = df_table.iloc[i]
	temp_his = np.zeros(venueSize)
	if len(loc_temp) > 0:
		oldRow = df.iloc[i - 1]
		if row['user_id'] == oldRow['user_id']:
			for j in range(len(loc_temp)):
				temp_his[loc_temp[j]] = float(loc_count_temp[j]) / sum(loc_count_temp)
		else:
			loc_temp = []
			loc_count_temp = []
	if row['train'] == 1:
		if loc_temp.count(row['loc_id']) > 0:
			loc_index = loc_temp.index(row['loc_id'])
			loc_count_temp[loc_index] += 1
		else:
			loc_temp.append(row['loc_id'])
			loc_count_temp.append(1)
	history[i] = temp_his
df_table['history'] = history
df_table['history'] = df_table['history'].apply(lambda x: np.array(x))
print("done")


def toSparseMatrix(x):
	I = range(K)
	J = np.full(K, x.loc_id)
	V = x.phi
	return sparse.coo_matrix((V, (I, J)), shape=(K, venueSize))


def updateRoleVenue(x):
	roleVenue = np.ones([K, len(venuelist)]) / len(venuelist)
	for index, row in x.iterrows():
		if row['train'] > 0:
			roleVenue[:, row['loc_id']] += row['phi']
	roleVenue /= roleVenue.sum(axis=1)[:, np.newaxis]
	return roleVenue


df_table['temp'] = df_table['loc_id'].apply(lambda x: roleVenue[:, x])


def f_test(x):
	temp_history = x[2]
	temp_loc = x[1]
	temp = x[0]
	temp[0] = temp_history[temp_loc]
	return temp


df_table['temp'] = df_table['loc_id'].apply(lambda x: roleVenue[:, x])


def multiplyArray(x):
	a = np.multiply(x[0], x[1])
	return a / (a.sum())


input_temp = df_table[['temp', 'gamma']].values
df_table['phi'] = np.apply_along_axis(multiplyArray, 1, input_temp).tolist()
df_table['phi'] = df_table['phi'].apply(lambda x: np.array(x))

roleVenue = updateRoleVenue(df_table)

print("optimize")
for i in range(50):
	print("iter " + str(i + 1))

	dfgamma = df_table.groupby(['user_id'])['phi'].apply(lambda x: x.sum() + np.ones(K) / K).to_frame().reset_index()
	dfgamma.columns = ['user_id', 'gamma']

	df_table['temp'] = df_table['loc_id'].apply(lambda x: roleVenue[:, x])

	start = time.time()
	input_temp = df_table[['temp', 'gamma']].values
	df_table['phi'] = np.apply_along_axis(multiplyArray, 1, input_temp).tolist()
	df_table['phi'] = df_table['phi'].apply(lambda x: np.array(x))
	print(time.time() - start)

	df_inference = df_table[['user_id', 'loc_id', 'phi', 'train', 'history']]

	df_table = pd.merge(df_inference, dfgamma, how='left', left_on=['user_id'], right_on=['user_id']) \
		[['user_id', 'loc_id', 'phi', 'gamma', 'train', 'history']]

	roleVenue = updateRoleVenue(df_table)


valid_user = df_table.groupby('user_id')['train'].mean()
valid_user = valid_user[(valid_user > 0.2) & (valid_user < 0.999)]
valid_user = list(set(valid_user.index))

user_prediction_count = np.zeros(len(nodelist))
user_true_prediction_count = np.zeros(len(nodelist))

test = pd.DataFrame(columns=['user_id', 'loc_id', 'repeated', 'rank'])
top_k = 100
ranking_avg = []
for index, row in df_table.iterrows():
	if row['train'] < 1 and row.user_id in valid_user:
		array = np.dot(row['gamma'] / sum(row['gamma']), roleVenue)
		order = array.argsort()
		ranks = venueSize - order.argsort()
		ranking_avg.append(ranks[row['loc_id']])
		if ranks[row['loc_id']] <= top_k:
			user_true_prediction_count[nodelist.index(row['user_id'])] += 1
		is_repeated = 0
		if row['history'][row['loc_id']] > 0:
			is_repeated = 1
		test.loc[len(ranking_avg)-1] = [row['user_id'], row['loc_id'], is_repeated, ranks[row['loc_id']]]
		user_prediction_count[nodelist.index(row['user_id'])] += 1

# print(reduce(lambda x, y: x + y, ranking_avg) / len(ranking_avg))
print(K)

a = []
for i in range(0, len(user_prediction_count)):
	if user_prediction_count[i] > 0:
		a.append(user_true_prediction_count[i] / user_prediction_count[i])
print(np.average(a))
print(sum(user_true_prediction_count) / sum(user_prediction_count))

repeated_true = test.loc[(test['repeated'] == 1) & (test['rank'] < top_k)]['user_id'].count()
repeated_false = test.loc[(test['repeated'] == 1) & (test['rank'] > top_k)]['user_id'].count()
print(repeated_true / (repeated_false + repeated_true))
new_true = test.loc[(test['repeated'] == 0) & (test['rank'] < top_k)]['user_id'].count()
new_false = test.loc[(test['repeated'] == 0) & (test['rank'] > top_k)]['user_id'].count()
print(new_true / (new_false + new_true))

# print(test)
