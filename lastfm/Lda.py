import pandas as pd
import numpy as np
import datetime
from scipy.special import digamma
import itertools
import random
from scipy import sparse
from functools import reduce

last_fm_sample = pd.read_csv("C:/Users/ducnm/Downloads/userid_artistid_time_sample.csv",
                             sep=',', header=0, index_col='Unnamed: 0')
K = 50

df_inference = last_fm_sample[['user_id', 'artist_id']]
temp = np.random.dirichlet(np.ones(K)*10,len(last_fm_sample)).tolist()
temparray = [np.array(row) for row in temp]
df_inference['phi'] =temparray

nodelist = list(set(df_inference.user_id.tolist()))
venuelist = list(set(df_inference.artist_id.tolist()))

venueSize = len(venuelist)

roleVenue = np.ones([K,len(venuelist)])/len(venuelist)

tempp = (np.random.dirichlet(np.ones(K)*100,len(nodelist))*K).tolist()
tempparray = [np.array(row) for row in tempp]
seriesgamma = tempparray
dfgamma = pd.DataFrame(list(zip(nodelist,seriesgamma)), columns=['user_id','gamma'])

df_table = pd.merge(df_inference, dfgamma, how='left', left_on=['user_id'],
                    right_on=['user_id'])[['user_id','artist_id','phi','gamma']]

df_table['train'] = 1
change = df_table.sample(int(df_table.shape[0]*0.1),random_state=123).index
df_table.loc[change,'train'] = 0

# df_table['history']= [[0]]*df_table.shape[0]
# df_table['history'] = df_table['history'].apply(lambda x: np.array(x))

print("calculate history")
history = [[0]] * df_table.shape[0]
loc_temp = []
loc_count_temp = []
for i in range(0, len(df_table)):
    row = df_table.iloc[i]
    temp_his = np.zeros(venueSize)
    if len(loc_temp) > 0:
        oldRow = last_fm_sample.iloc[i-1]
        if row['user_id'] == oldRow['user_id']:
            for j in range(len(loc_temp)):
                temp_his[loc_temp[j]] = float(loc_count_temp[j]) / sum(loc_count_temp)
        else:
            loc_temp = []
            loc_count_temp = []
    if row['train'] == 1:
        if loc_temp.count(row['artist_id']) > 0:
            loc_index = loc_temp.index(row['artist_id'])
            loc_count_temp[loc_index] += 1
        else:
            loc_temp.append(row['artist_id'])
            loc_count_temp.append(1)
    history[i] = temp_his
df_table['history'] = history
df_table['history'] = df_table['history'].apply(lambda x: np.array(x))
print("done")
# print(df_table)


def toSparseMatrix(x):
    I = range(K)
    J = np.full(K, x.artist_id)
    V = x.phi
    return sparse.coo_matrix((V, (I, J)), shape=(K, venueSize))


def updateRoleVenue(x):
    roleVenue = np.ones([K, len(venuelist)]) / len(venuelist)
    for index, row in x.iterrows():
        if row['train'] > 0:
            roleVenue[:, row['artist_id']] += row['phi']
    roleVenue /= roleVenue.sum(axis=1)[:, np.newaxis]
    return roleVenue


df_table['temp'] = df_table['artist_id'].apply(lambda x: roleVenue[:, x])


def f_test(x):
    temp_history = x[2]
    temp_loc = x[1]
    temp = np.append(np.array(temp_history[temp_loc]), x[0][1:K])
    return temp


df_table['temp'] = df_table[['temp', 'artist_id', 'history']].apply(f_test, axis=1)


def multiplyArray(x):
    return tuple(np.multiply(x['temp'], x['gamma']))


df_table['phi'] = df_table.apply(multiplyArray, axis=1) \
    .apply(np.array).apply(lambda x: x / x.sum())

roleVenue = updateRoleVenue(df_table)

print("optimize")
for i in range(50):
    print("iter " + str(i+1))
    # df_table = df.groupby(['user_id']).apply(updateGamma)
    dfgamma = df_table.groupby(['user_id'])['phi'].apply(lambda x: x.sum() + np.ones(K) / K).to_frame().reset_index()
    dfgamma.columns = ['user_id', 'gamma']

    df_table['temp'] = df_table['artist_id'].apply(lambda x: roleVenue[:, x])
    df_table['temp'] = df_table[['temp', 'artist_id', 'history']].apply(f_test, axis=1)
    df_table['phi'] = df_table.apply(multiplyArray, axis=1) \
        .apply(np.array).apply(lambda x: x / x.sum())

    df_inference = df_table[['user_id', 'artist_id', 'phi', 'train', 'history']]
    df_table = pd.merge(df_inference, dfgamma, how='left', left_on=['user_id'], right_on=['user_id']) \
        [['user_id', 'artist_id', 'phi', 'gamma', 'train', 'history']]
    roleVenue = updateRoleVenue(df_table)


def f_test2(x):
    gamma = x[0]
    temp_history = x[2]
    roleVenue2 = np.array(roleVenue)
    roleVenue2[0] = temp_history
    temp2 = np.dot(gamma / sum(gamma), roleVenue2)
    return temp2


user_prediction_count = np.zeros(len(nodelist))
user_true_prediction_count = np.zeros(len(nodelist))
true_prediction_count = 0
prediction_count = 0
top_k = 100
ranking_avg = []
for index, row in df_table.iterrows():
    if row['train'] < 1:
        roleVenue2 = np.array(roleVenue)
        roleVenue2[0] = row['history']
        array = np.dot(row['gamma'] / sum(row['gamma']), roleVenue2)
        order = array.argsort()
        ranks = venueSize - order.argsort()
        ranking_avg.append(ranks[row['artist_id']])


# print(reduce(lambda x, y: x + y, ranking_avg) / len(ranking_avg))
print(K)
print(float(true_prediction_count) / prediction_count)

a = []
for i in range(0, len(user_prediction_count)):
    if user_prediction_count[i] > 0:
        a.append(user_true_prediction_count[i] / user_prediction_count[i])
print(np.average(a))

