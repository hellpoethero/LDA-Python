import pandas as pd
import numpy as np


datasetname = "go_sf"
sss = [datasetname+"_K2_I1",
       datasetname+"_K3_I100",
       datasetname+"_K4_I100",
       datasetname+"_K5_I100",
       datasetname+"_K10_I100"]

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
	df = df.sample(frac=0.5, replace=True, random_state=1)
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

loc_user_dict = df[df['train'] == 1].groupby('user_id')['loc_id'].apply(list).to_dict()
repeated_array = np.zeros(len(df))

for r in zip(df['user_id'], df['loc_id']):
	print(loc_user_dict[r[0]])

index = 0
for filename in sss:
	full_path = "D:/Research/Project/LDA/results/" + filename + ".csv"
	ranks = pd.read_csv(full_path, index_col=0)
	max_rank = max(ranks['rank'].values)
	num_record = len(ranks['rank'])
	# print(max_rank)

	unique, counts = np.unique(ranks['rank'].values, return_counts=True)
	cum_counts = np.cumsum(counts) / num_record

	auc = 0.0
	prev_value = [0, 0]
	for value in zip(unique / max_rank, cum_counts):
		if prev_value[0] != 0:
			s = (value[1] + prev_value[1]) / 2 * (value[0] - prev_value[0])
			auc += s
		prev_value = value

	print(sss[index] + "\t" + str([auc, np.mean(ranks['rank'].values)]))
	index += 1
