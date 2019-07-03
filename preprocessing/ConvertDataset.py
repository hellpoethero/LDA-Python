import datetime
import pandas as pd
import numpy as np
import random
from scipy import sparse

dataset = "go_ny"
trainFile = "D:/Download/data/"+dataset+"/train.csv"
validationFile = "D:/Download/data/"+dataset+"/validation.csv"
testFile = "D:/Download/data/"+dataset+"/test.csv"
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
	# df = df.sample(frac=0.01, replace=True, random_state=1)
	df.columns = ["user_id", "loc_id", "train"]
	# print(df.groupby(['user_id', 'loc_id']).size().reset_index(name='counts'))
	# df.groupby(['user_id', 'loc_id']).count().to_csv("D:/Download/data/last_fm_train_190617.csv", header=None)
	return df


def function02(data):
	user_number = data['user_id'].max() + 1
	user_array = []
	for i in range(0, user_number):
		user_array.append([])
	x1 = data.groupby(['user_id'])
	x1_list = x1['loc_id'].apply(list)
	for key in x1.groups.keys():
		user_array[key] = x1_list[key]
		# random.shuffle(user_array[key])
		print(key, end=" ")
		print(len(user_array[key]))

	# print(user_array)


def function03(data):
	count = 0
	user_array = []
	user_array.append([])
	last_user_id = data.iloc[0]['user_id']
	for index, row in data.iterrows():
		a = str(row['loc_id']) + ":" + str(row['count'])
		if row['user_id'] != last_user_id:
			count += 1
			user_array.append([])
		user_array[count].append(a)
		last_user_id = row['user_id']
	# print(user_array)
	return user_array


df_table = pd.concat([function01(train, 1), function01(validation, 1), function01(test, 0)])
df_table.columns = ['user_id', 'loc_id', 'train']
valid_user = df_table.groupby('user_id')['train'].mean()
valid_user = valid_user[(valid_user > 0.001) & (valid_user < 0.999)]
valid_user = list(set(valid_user.index))
df_table = df_table[df_table['user_id'].isin(valid_user)]

train_df = df_table[df_table['train'] == 1]
train_df = train_df.groupby(['user_id', 'loc_id']).size().reset_index(name='count')
test_df = df_table[df_table['train'] == 0]
test_df = test_df.groupby(['user_id', 'loc_id']).size().reset_index(name='count')

# user_id_df = pd.DataFrame(train_df['loc_id'].unique())
# user_id_df['index1'] = user_id_df.index
# user_id_df.columns = ['loc_id', 'index1']
#
# train_df = pd.merge(train_df, user_id_df, how='inner', left_on=['loc_id'], right_on=['loc_id']) \
# 	[['user_id', 'index1', 'count']]
# train_df.columns = ['user_id', 'loc_id', 'count']

# print(train_df['user_id'].nunique())
# print(test_df['user_id'].nunique())
train_str_array = function03(train_df)
test_str_array = function03(test_df)
print(len(train_str_array))
print(len(test_str_array))
#
with open("D:/Research/Dataset/checkin/ny_190617/train.dat", "w") as trainFile:
	for user in train_str_array:
		trainFile.write(str(len(user)) + " " + " ".join(user) + "\n")

with open("D:/Research/Dataset/checkin/ny_190617/validation.dat", "w") as testFile:
	for user in test_str_array:
		testFile.write(str(len(user)) + " " + " ".join(user) + "\n")

# function02(train_df)

# c = pd.concat([pd.DataFrame(a).sample(frac=0.01, replace=True, random_state=1),
# 			   pd.DataFrame(b).sample(frac=0.01, replace=True, random_state=1)]).sort_values(by=[0, 2], ascending=[1, 0])
# c.columns = ['user_id', 'loc_id', 'train']
#
# loc_df = pd.DataFrame(c['loc_id'].unique())
# loc_df['index1'] = loc_df.index
# loc_df.columns = ['loc_id', 'index1']
#
# df = pd.merge(c, loc_df, how='inner', left_on=['loc_id'], right_on=['loc_id']) \
# 	[['user_id', 'index1', 'train']]
# df.columns = ['user_id', 'loc_id', 'train']
# df = df.sort_values(by=['user_id', 'train'], ascending=[1, 0])
#
# print(df)
# print(len(df))
# df.to_csv("D:/Download/data/last_fm_190617.csv")
