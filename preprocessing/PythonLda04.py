import datetime
import pandas as pd
import numpy as np
import random
from scipy import sparse
from scipy.special import digamma
from scipy.sparse import csr_matrix
from time import time
# from guppy import hpy

path = "D:/Download/data/"
# datasetName = "lastfm_sample_190617/"
datasetName = "lastfm/"
# "/Users/yoonsik/Downloads/
trainFile = path + datasetName + "train.csv"
validationFile = path + datasetName + "validation.csv"
testFile = path + datasetName + "test.csv"
train = pd.read_csv(trainFile, header=None)
validation = pd.read_csv(validationFile, header=None)
test = pd.read_csv(testFile, header=None)

train.columns = ["user_id", "loc_id", "count"]
validation.columns = ["user_id", "loc_id", "count"]
test.columns = ["user_id", "loc_id", "count"]


def function01(data, train):
	array = []
	for index, row in data.iterrows():
		temp = [[row['user_id'], row['loc_id'], train]] * row['count']
		array.extend(temp)

	df = pd.DataFrame(array)
	df = df.sample(frac=0.3, replace=True, random_state=1)
	df.columns = ["user_id", "loc_id", "train"]
	return df


train_df = pd.concat([function01(train, 1), function01(validation, 1)])
test_df = function01(test, 0)
test_df = test_df[test_df['loc_id'].isin(train_df['loc_id'].tolist())]

df = pd.concat([train_df, test_df])

loc_df = pd.DataFrame(df['loc_id'].unique())
loc_df['index1'] = loc_df.index
loc_df.columns = ['loc_id', 'index1']

df = pd.merge(df, loc_df, how='inner', left_on=['loc_id'], right_on=['loc_id']) \
	[['user_id', 'index1', 'train']]
df.columns = ['user_id', 'loc_id', 'train']
df = df.sort_values(by=['user_id', 'train'], ascending=[1, 0])

num_record = len(df)
print(num_record )

K = 2

df_inference = df[['user_id', 'loc_id', 'train']]
temp = np.random.dirichlet(np.ones(K) * 1000, len(df)).tolist()
temparray = [np.array(row) for row in temp]
df_inference['phi'] = temparray

nodelist = list(set(df_inference.user_id.tolist()))
venuelist = list(set(df_inference.loc_id.tolist()))

venueSize = len(venuelist)

print(venueSize)
print(len(nodelist))

roleVenue = np.ones([K, len(venuelist)]) / len(venuelist)

tempp = (np.random.dirichlet(np.ones(K) * 100, len(nodelist)) * K).tolist()
tempparray = [np.array(row) for row in tempp]
seriesgamma = tempparray
dfgamma = pd.DataFrame(list(zip(nodelist, seriesgamma)), columns=['user_id', 'gamma'])

df_table = pd.merge(df_inference, dfgamma, how='left', left_on=['user_id'], right_on=['user_id']) \
	[['user_id', 'loc_id', 'phi', 'train', 'gamma']]

# df_table['history'] = [[0]] * df_table.shape[0]
# df_table['history'] = df_table['history'].apply(lambda x: np.array(x))

print("Calculate the history")
history = [[0]] * num_record
start = time()
temp_his = np.zeros(venueSize)
# temp_his = csr_matrix((1,venueSize))
temp_count = 0
index = 0
# print(df_table)
# for row in zip(df_table['user_id'], df_table['loc_id'], df['train']):
# # for index, row in df_table.iterrows():
# # 	if index == 0 or row['user_id'] != df.iloc[index - 1]['user_id']:
# 	print(index)
# 	if index == 0 or row[0] != df.iloc[index - 1]['user_id']:
# 		temp_his = np.zeros(venueSize)
# 		# temp_his = csr_matrix((1,venueSize))
# 		temp_count = 0
# 		history[index] = csr_matrix(temp_his)
# 		# history[index] = temp_his
# 	else:
# 		history[index] = csr_matrix(temp_his / temp_count)
# 		# history[index] = temp_his/sum(temp_his)
# 	# if row['train'] == 1:
# 	if row[2] == 1:
# 		# temp_his[row['loc_id']] += 1
# 		temp_his[row[1]] += 1
# 		# temp_his[(0, row['loc_id'])] += 1
# 		temp_count += 1
# 	index += 1

# print h.heap()

repeated = np.zeros(num_record )
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

# df_table['temp'] = df_table['loc_id'].apply(lambda x: csr_matrix(([1.0], ([0], [x])), shape=(1, venueSize)))
# df_table['temp'] = df_table.groupby(['user_id'])['temp'].shift(1).fillna(0)
# df_table['temp'] = df_table.groupby(['user_id'])['temp'].apply(lambda x: x.cumsum())


# def function02(x):
# 	x_sum = 0
# 	for value in x:
# 		x_sum = sum(value[0])
# 	x = x / x_sum
# 	print(x)
#
#
# df_table.loc[9:9, ['temp']].apply(lambda x: function02(x))

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
# 	print(df_table.loc[4:4,['temp']])
# 	print(df_table.loc[5:5,['temp']])
# 	print(df_table.loc[6:6,['temp']])
# print(df_table['temp'])
print(time() - start)
print("done")
# print(df_table['history'])

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
	roleVenue = np.ones([K, len(venuelist)]) / len(venuelist)

	a = x.groupby('loc_id')['phi'].apply(lambda x: x.sum()).sort_index()
	b = np.array(a.tolist())
	b = b.T
	# roleVenue += b
	roleVenue = b
	roleVenue /= roleVenue.sum(axis=1)[:, np.newaxis]

	roleVenue = 0.05 * np.array([g, ] * K) + 0.95 * roleVenue
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
g = df_table.groupby('loc_id')['phi'].apply(lambda x: x.count()).sort_index()
g = np.array(g.tolist())
g = g.T
g = g / float(sum(g))

print("optimize")
for i in range(1):
	print("iter " + str(i + 1))

	dfgamma = df_table.groupby(['user_id'])['phi'].apply(lambda x: x.sum() + np.ones(K) / K).to_frame().reset_index()
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
valid_user = df_table_backup.groupby('user_id')['train'].mean()
valid_user = valid_user[(valid_user > 0.001) & (valid_user < 0.999)]
valid_user = list(set(valid_user.index))

user_prediction_count = np.zeros(len(nodelist))
user_true_prediction_count = np.zeros(len(nodelist))

test = pd.DataFrame(columns=['user_id', 'loc_id', 'repeated', 'rank'])
top_k = 100
ranking_avg = []

beta_0 = 0
beta_1 = 0
# for index, row in df_table.iterrows():
#     if row['train'] > 0 and row.user_id in valid_user:
#         beta_0 += row['gamma'][0]
#         beta_1 += row['gamma'][1:].sum()

df_inference_testing = df_table_testing[['user_id', 'loc_id', 'phi', 'train', 'history', 'repeated']]
df_table_testing = pd.merge(df_inference_testing, dfgamma, how='left', left_on=['user_id'], right_on=['user_id']) \
	[['user_id', 'loc_id', 'phi', 'gamma', 'train', 'history', 'repeated']]

df_table_testing = df_table_testing[df_table_testing['user_id'].isin(valid_user)].reset_index()
in_top = np.zeros(len(df_table_testing))
print("calculate ranking")
for index, row in df_table_testing.iterrows():
	roleVenue2 = np.array(roleVenue)
	roleVenue2[0] = row['history'].toarray()
	array = np.dot(row['gamma'] / sum(row['gamma']), roleVenue2)
	order = array.argsort()
	ranks = venueSize - order.argsort()
	if ranks[row['loc_id']] <= top_k:
		# user_true_prediction_count[nodelist.index(row['user_id'])] += 1
		in_top[index] = 1
	# is_repeated = 0
	# if row['history'][(0, row['loc_id'])] > 0:
	# 	is_repeated = 1
	# test.loc[len(ranking_avg) - 1] = [row['user_id'], row['loc_id'], is_repeated, ranks[row['loc_id']]]
	# user_prediction_count[nodelist.index(row['user_id'])] += 1


# print(reduce(lambda x, y: x + y, ranking_avg) / len(ranking_avg))
print(K)
df_table_testing['in_top'] = in_top

# a = []
# for i in range(0, len(user_prediction_count)):
# 	if user_prediction_count[i] > 0:
# 		a.append(user_true_prediction_count[i] / user_prediction_count[i])
# print(np.average(a))
# print(sum(user_true_prediction_count) / sum(user_prediction_count))

print(np.mean(df_table_testing.groupby('user_id')['in_top'].mean().values))
print(sum(df_table_testing['in_top']) / len(df_table_testing))

print(np.mean(df_table_testing[df_table_testing['repeated'] == 1].groupby('user_id')['in_top'].mean().values))
print(np.mean(df_table_testing[df_table_testing['repeated'] == 0].groupby('user_id')['in_top'].mean().values))
# repeated_true = test.loc[(test['repeated'] == 1) & (test['rank'] < top_k)]['user_id'].count()
# repeated_false = test.loc[(test['repeated'] == 1) & (test['rank'] > top_k)]['user_id'].count()
# print(repeated_true / (repeated_false + repeated_true))
# new_true = test.loc[(test['repeated'] == 0) & (test['rank'] < top_k)]['user_id'].count()
# new_false = test.loc[(test['repeated'] == 0) & (test['rank'] > top_k)]['user_id'].count()
# print(new_true / (new_false + new_true))
