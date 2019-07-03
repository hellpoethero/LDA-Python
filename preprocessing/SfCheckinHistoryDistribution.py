import pandas as pd
import numpy as np
import datetime
from scipy.special import digamma
import itertools
import random
from scipy import sparse
from functools import reduce

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
df_table['history'] = [[0]] * df_table.shape[0]

df_table['history'] = df_table['history'].apply(lambda x: np.array(x))

history = [[0]] * df_table.shape[0]
loc_temp = []
loc_count_temp = []
for i in range(0, len(df_table)):
	row = df_table.iloc[i]
	temp_his = np.zeros(venueSize)
	if row['train'] == 1:
		if len(loc_temp) > 0:
			oldRow = df.iloc[i - 1]
			if row['user_id'] == oldRow['user_id']:
				for j in range(len(loc_temp)):
					temp_his[loc_temp[j]] = float(loc_count_temp[j]) / sum(loc_count_temp)
			else:
				loc_temp = []
				loc_count_temp = []
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

print(sum(df_table.iloc[10]['history']))
