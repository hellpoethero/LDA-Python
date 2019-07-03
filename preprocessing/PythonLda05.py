import datetime
import pandas as pd
import numpy as np
import random
from scipy import sparse
from scipy.special import digamma
from scipy.sparse import csr_matrix
from time import time
from functools import reduce
# from guppy import hpy

K = 5
path = "D:/Download/data/"
# datasetName = "lastfm_sample_190617/"
datasetName = "reddit_sample/"
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
	# df = df.sample(frac=0.4, replace=True, random_state=1)
	df.columns = ["user_id", "loc_id", "train"]
	return df


train_df = pd.concat([function01(train, 1), function01(validation, 1)])
test_df = function01(test, 0)
test_df = test_df[test_df['loc_id'].isin(train_df['loc_id'].tolist())]

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
df = df.sample(frac=1, random_state=123).sort_values(by=['user_id', 'train'], ascending=[1, 0])

# print(df[df['user_id'] == 100])
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


print(len(roleVenue[0]))
print("optimize")
numIter = 50
for i in range(numIter):
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


print(len(roleVenue[0]))
print("Optimize done")
valid_user = df_table_backup.groupby('user_id')['train'].mean()
valid_user = valid_user[(valid_user > 0.001) & (valid_user < 0.999)]
valid_user = list(set(valid_user.index))

test = pd.DataFrame(columns=['user_id', 'loc_id', 'repeated', 'rank'])
top_k = 100

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
		# user_true_prediction_count[nodelist.index(row['user_id'])] += 1
		in_top[index] = 1
	ranking_avg[index] = ranks[row['loc_id']]
	# is_repeated = 0
	# if row['history'][(0, row['loc_id'])] > 0:
	# 	is_repeated = 1
	# test.loc[len(ranking_avg) - 1] = [row['user_id'], row['loc_id'], is_repeated, ranks[row['loc_id']]]
	# user_prediction_count[nodelist.index(row['user_id'])] += 1


print(reduce(lambda x, y: x + y, ranking_avg) / len(ranking_avg))
print(K)
df_table_testing['in_top'] = in_top
df_table_testing['rank'] = ranking_avg

print(np.mean(df_table_testing.groupby('user_id')['in_top'].mean().values))
print(sum(df_table_testing['in_top']) / len(df_table_testing))

print(np.mean(df_table_testing[df_table_testing['repeated'] == 1].groupby('user_id')['in_top'].mean().values))
print(np.mean(df_table_testing[df_table_testing['repeated'] == 0].groupby('user_id')['in_top'].mean().values))

df_table_testing[['user_id', 'loc_id', 'rank']].to_csv(
	"D:/Research/Project/LDA/results/"+datasetName[:-1]+"_K"+str(K)+"_I"+str(numIter)+".csv")

