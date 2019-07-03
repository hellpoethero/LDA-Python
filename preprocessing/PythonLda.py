import pandas as pd
import numpy as np
import datetime
from scipy.special import digamma
import itertools
import random
from scipy import sparse
from functools import reduce
import time
import swifter

K = 50
file_path = "C:/Users/ducnm/Downloads/SFcheckin_0509_2.csv"
df = pd.read_csv(file_path,sep=',', index_col=0)
df_inference = df[['user_id', 'loc_id']]
temp = np.random.dirichlet(np.ones(K)*10, len(df)).tolist()
#temp = (np.ones((len(df),K))/K).tolist()
temparray = [np.array(row) for row in temp]
df_inference['phi'] = temparray

nodelist = list(set(df_inference.user_id.tolist()))
venuelist = list(set(df_inference.loc_id.tolist()))

venueSize = len(venuelist)

roleVenue = np.ones([K, len(venuelist)])/len(venuelist)

tempp = (np.random.dirichlet(np.ones(K)*100, len(nodelist))*K).tolist()
tempparray = [np.array(row) for row in tempp]
seriesgamma = tempparray
dfgamma = pd.DataFrame(list(zip(nodelist,seriesgamma)), columns=['user_id', 'gamma'])

df_table = pd.merge(df_inference, dfgamma, how='left', left_on=['user_id'], right_on=['user_id'])\
                [['user_id', 'loc_id', 'phi', 'gamma']]

df_table['train'] = 1
change = df_table.sample(int(df_table.shape[0]*0.1), random_state=123).index
df_table.loc[change, 'train'] = 0
df_table['history'] = [[0]]*df_table.shape[0]
df_table['history'] = df_table['history'].apply(lambda x: np.array(x))

history = [[0]] * df_table.shape[0]
loc_temp = []
loc_count_temp = []
for i in range(0, len(df_table)):
    row = df_table.iloc[i]
    temp_his = np.zeros(venueSize)
    if len(loc_temp) > 0:
        oldRow = df.iloc[i-1]
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
    #temp = np.append(np.array(temp_history[temp_loc]), x[0][1:K])
    return temp


df_table['temp'] = df_table[['temp', 'loc_id', 'history']].apply(f_test, axis=1)


def multiplyArray_backup(x):
    return tuple(np.multiply(x['temp'], x['gamma']))

def multiplyArray(x):
    a =  np.multiply(x[0],x[1])
    return a/(a.sum())

# input_temp = df_table[['temp', 'gamma']].values
# df_table['phi'] = np.apply_along_axis(multiplyArray, 1, input_temp)
df_table['phi'] = df_table.apply(multiplyArray_backup, axis=1) \
    .apply(np.array).apply(lambda x: x / x.sum())

roleVenue = updateRoleVenue(df_table)


vf_test = np.vectorize(f_test)


print("optimize")
for i in range(10):
    print("iter " + str(i+1))
    # df_table = df.groupby(['user_id']).apply(updateGamma)
    start_time = time.time()
    dfgamma = df_table.groupby(['user_id'])['phi'].apply(lambda x: x.sum() + np.ones(K) / K).to_frame().reset_index()
    elapsed_time = time.time() - start_time
    # print(elapsed_time)
    dfgamma.columns = ['user_id', 'gamma']

    start_time = time.time()
    df_table['temp'] = df_table['loc_id'].apply(lambda x: roleVenue[:, x])
    elapsed_time = time.time() - start_time
    print(elapsed_time)

    start_time = time.time()
#    data_temp_1 = df['temp'].values  # Assuming you have two columns A and B
#    data_temp_2 = df['loc_id'].values
#    data_temp_3 = df['history'].values
#    input_temp = np.concatenate((data_temp_1,data_temp_2,data_temp_3), axis=0)
#     input_temp = df_table[['temp', 'loc_id', 'history']].values
#     df_table['temp'] = np.apply_along_axis(f_test, 1, input_temp)

    df_table['temp'] = df_table[['temp', 'loc_id', 'history']].apply(f_test, axis=1)
    elapsed_time = time.time() - start_time
    print(elapsed_time)

    start_time = time.time()

    input_temp = df_table[['temp', 'gamma']].values
    # df_table['phi'] = np.apply_along_axis(multiplyArray,1, input_temp)
    df_table['phi'] = df_table.apply(multiplyArray_backup, axis=1) \
       .apply(np.array).apply(lambda x: x / x.sum())

    elapsed_time = time.time() - start_time
    print(elapsed_time)

    df_inference = df_table[['user_id', 'loc_id', 'phi', 'train', 'history']]
    start_time = time.time()

    df_table = pd.merge(df_inference, dfgamma, how='left', left_on=['user_id'], right_on=['user_id']) \
        [['user_id', 'loc_id', 'phi', 'gamma', 'train', 'history']]
    elapsed_time = time.time() - start_time
    # print(elapsed_time)
    start_time = time.time()

    roleVenue = updateRoleVenue(df_table)
    elapsed_time = time.time() - start_time
    # print(elapsed_time)


def f_test2(x):
    gamma = x[0]
    temp_history = x[2]
    roleVenue2 = np.array(roleVenue)
    roleVenue2[0] = temp_history
    temp2 = np.dot(gamma / sum(gamma), roleVenue2)
    return temp2


user_prediction_count = np.zeros(len(nodelist))
user_true_prediction_count = np.zeros(len(nodelist))

# print(len(nodelist))
# print(nodelist)

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
        ranking_avg.append(ranks[row['loc_id']])
        if ranks[row['loc_id']] <= top_k:
            true_prediction_count += 1
            user_true_prediction_count[nodelist.index(row['user_id'])] += 1
        prediction_count += 1
        user_prediction_count[nodelist.index(row['user_id'])] += 1

# print(reduce(lambda x, y: x + y, ranking_avg) / len(ranking_avg))
print(K)
print(float(true_prediction_count) / prediction_count)

a = []
for i in range(0, len(user_prediction_count)):
    if user_prediction_count[i] > 0:
        a.append(user_true_prediction_count[i] / user_prediction_count[i])
print(np.average(a))
