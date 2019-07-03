import pandas as pd
import numpy as np
import datetime
from scipy.special import digamma
import itertools
import random
from scipy import sparse
from functools import reduce

last_fm_sample = pd.read_csv("C:/Users/ducnm/Downloads/userid_artistid_time_sample.csv", sep=',', header=0, index_col='Unnamed: 0')
K = 500

df_inference = last_fm_sample[['user_id', 'artist_id']]
temp = np.random.dirichlet(np.ones(K)*10,len(last_fm_sample)).tolist()
temparray = [np.array(row) for row in temp]
df_inference['phi']=temparray

nodelist = list(set(df_inference.user_id.tolist()))
venuelist = list(set(df_inference.artist_id.tolist()))

venueSize = len(venuelist)

roleVenue = np.ones([K,len(venuelist)])/len(venuelist)

tempp = (np.random.dirichlet(np.ones(K)*100,len(nodelist))*K).tolist()
tempparray = [np.array(row) for row in tempp]
seriesgamma = tempparray
dfgamma = pd.DataFrame(list(zip(nodelist,seriesgamma)), columns=['user_id','gamma'])

df_table = pd.merge(df_inference, dfgamma, how='left', left_on=['user_id'], right_on=['user_id'])\
                [['user_id','artist_id','phi','gamma']]

df_table['train'] = 1
change = df_table.sample(int(df_table.shape[0]*0.1),random_state=123).index
df_table.loc[change,'train'] = 0


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
    df_table['phi'] = df_table.apply(multiplyArray, axis=1) \
        .apply(np.array).apply(lambda x: x / x.sum())

    df_inference = df_table[['user_id', 'artist_id', 'phi', 'train']]
    df_table = pd.merge(df_inference, dfgamma, how='left', left_on=['user_id'], right_on=['user_id']) \
        [['user_id', 'artist_id', 'phi', 'gamma', 'train']]
    roleVenue = updateRoleVenue(df_table)


ranking_avg = []
for index, row in df_table.iterrows():
    if row['train'] < 1:
        array = np.dot(row['gamma'] / sum(row['gamma']), roleVenue)
        order = array.argsort()
        ranks = venueSize - order.argsort()
        ranking_avg.append(ranks[row['artist_id']])


print(reduce(lambda x, y: x + y, ranking_avg) / len(ranking_avg))
print(K)
# print(df_table['ttttt'])
