import datetime
import pandas as pd
import numpy as np
import random
from scipy import sparse


datasetName = "go_sf"
trainFile = "D:/Download/data/"+datasetName+"/train.csv"
validationFile = "D:/Download/data/"+datasetName+"/validation.csv"
testFile = "D:/Download/data/"+datasetName+"/test.csv"
train = pd.read_csv(trainFile, header=None)
validation = pd.read_csv(validationFile, header=None)
test = pd.read_csv(testFile, header=None)

train.columns = ["user_id", "loc_id", "count"]
validation.columns = ["user_id", "loc_id", "count"]
test.columns = ["user_id", "loc_id", "count"]


train_loc = set(train.loc_id.tolist())
valid_loc = set(validation.loc_id.tolist())
test_loc = set(test.loc_id.tolist())

a = []
for index, row in train.iterrows():
	for count in range(0, row['count']):
		a.append([row['user_id'], row['loc_id'], 1])

for index, row in validation.iterrows():
	for count in range(0, row['count']):
		a.append([row['user_id'], row['loc_id'], 1])
random.shuffle(a)
a_df = pd.DataFrame(a)
a_df.columns = ['user_id', 'loc_id', 'train']
print(len(set(a_df['loc_id'].tolist())))

test = test[test['loc_id'].isin(a_df['loc_id'].tolist())]

b = []
for index, row in test.iterrows():
	for count in range(0, row['count']):
		b.append([row['user_id'], row['loc_id'], 0])
random.shuffle(b)

c = pd.concat([pd.DataFrame(a), pd.DataFrame(b)])
c.columns = ['user_id', 'loc_id', 'train']

print(len(set(c['loc_id'].tolist())))

loc_df = pd.DataFrame(c['loc_id'].unique())
loc_df['index1'] = loc_df.index
loc_df.columns = ['loc_id', 'index1']

df = pd.merge(c, loc_df, how='inner', left_on=['loc_id'], right_on=['loc_id']) \
	[['user_id', 'index1', 'train']]
df.columns = ['user_id', 'loc_id', 'train']
df = df.sort_values(by=['user_id', 'train'], ascending=[1, 0])

K = 100

df_inference = df[['user_id', 'loc_id', 'train']]
temp = np.random.dirichlet(np.ones(K) * 10, len(df)).tolist()
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
# df_table['history'] = df_table['history'].apply(lambda x: np.array(x))
print("done")

df_table_training = df_table[df_table.train==1]

print(len(set(df_table[df_table.train==1]['loc_id'].tolist())))
df_table_testing = df_table[df_table.train==0]
df_table_backup = df_table
df_table = df_table_training
def toSparseMatrix(x):
	I = range(K)
	J = np.full(K, x.loc_id)
	V = x.phi
	return sparse.coo_matrix((V, (I, J)), shape=(K, venueSize))

#
# def updateRoleVenue(x):
# 	roleVenue = np.ones([K, len(venuelist)]) / len(venuelist)
# 	for index, row in x.iterrows():
# 		if row['train'] > 0:
# 			roleVenue[:, row['loc_id']] += row['phi']
# 	roleVenue /= roleVenue.sum(axis=1)[:, np.newaxis]
# 	return roleVenue


def updateRoleVenue(x):
	roleVenue = np.ones([K, len(venuelist)]) / len(venuelist)

	a = x.groupby('loc_id')['phi'].apply(lambda x: x.sum()).sort_index()
	# print(len(set(x['loc_id'].tolist())))
	b = np.array(a.tolist())
	# print(b.shape)
	b = b.T
	roleVenue += b
	# roleVenue = np.ones([K, len(venuelist)]) / len(venuelist)
	# for index, row in x.iterrows():
	#    if row['train'] > 0:
	#        roleVenue[:, row['loc_id']] += row['phi']
	roleVenue /= roleVenue.sum(axis=1)[:, np.newaxis]
	return roleVenue

df_table['temp'] = df_table['loc_id'].apply(lambda x: roleVenue[:, x])


def f_test(x):
	temp_history = x[2]
	temp_loc = x[1]
	temp = x[0]
	temp[0] = temp_history[temp_loc]
	return temp


input_temp = df_table[['temp', 'loc_id', 'history']].values
df_table['temp'] = np.apply_along_axis(f_test, 1, input_temp).tolist()
df_table['temp'] = df_table['temp'].apply(lambda x: np.array(x))


def multiplyArray_backup(x):
	return tuple(np.multiply(x['temp'], x['gamma']))


def multiplyArray(x):
	a = np.multiply(x[0], x[1])
	return a / (a.sum())


input_temp = df_table[['temp', 'gamma']].values
df_table['phi'] = np.apply_along_axis(multiplyArray, 1, input_temp).tolist()
df_table['phi'] = df_table['phi'].apply(lambda x: np.array(x))

roleVenue = updateRoleVenue(df_table)

vf_test = np.vectorize(f_test)

print("optimize")
for i in range(50):
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


valid_user = df_table_backup.groupby('user_id')['train'].mean()
valid_user = valid_user[(valid_user > 0.2) & (valid_user < 0.999)]
valid_user = list(set(valid_user.index))

user_prediction_count = np.zeros(len(nodelist))
user_true_prediction_count = np.zeros(len(nodelist))

test = pd.DataFrame(columns=['user_id', 'loc_id', 'repeated', 'rank'])
top_k = 100
ranking_avg = []

beta_0 = 0
beta_1 = 0
# for index, row in df_table.iterrows():
# 	if row['train'] > 0 and row.user_id in valid_user:
# 		beta_0 += row['gamma'][0]
# 		beta_1 += row['gamma'][1:].sum()

df_inference_testing = df_table_testing[['user_id', 'loc_id', 'phi', 'train', 'history']]
df_table_testing = pd.merge(df_inference_testing, dfgamma, how='left', left_on=['user_id'], right_on=['user_id']) \
		[['user_id', 'loc_id', 'phi', 'gamma', 'train', 'history']]

for index, row in df_table_testing.iterrows():
	if row['train'] < 1 and row.user_id in valid_user:
		roleVenue2 = np.array(roleVenue)
		roleVenue2[0] = row['history']
		# temp_prob_topic = row['gamma'][1:]
		# temp_prob_topic = temp_prob_topic/sum(temp_prob_topic)
		# temp_prob = np.zeros(K)
		# temp_prob[0] = beta_0/(beta_0+beta_1)
		# temp_prob[1:] = beta_1/(beta_0 + beta_1)*temp_prob_topic
		# array = np.dot(temp_prob, roleVenue2)
		array = np.dot(row['gamma'] / sum(row['gamma']), roleVenue2)
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
