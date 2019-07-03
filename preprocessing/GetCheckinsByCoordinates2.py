import datetime
import pandas as pd
import numpy as np

import random

file_path_checkin = "D:\Research\Dataset\Gowalla_totalCheckins.txt"
df_checkin = pd.read_csv(file_path_checkin, sep='\t', header=None)
df_checkin.columns = ['user_id', 'time', 'lat', 'long', 'loc_id']

# filtereddf_checkin = df_checkin[(df_checkin.lat > 37.643083)
# 								& (df_checkin.lat < 37.833210)
# 								& (df_checkin.long > -122.555100)
# 								& (df_checkin.long < -122.355660)]
filtereddf_checkin = df_checkin[(df_checkin.lat > 40.569448)
								& (df_checkin.lat < 40.915483)
								& (df_checkin.long > -74.011864)
								& (df_checkin.long < -73.701145)]

# filtereddf_checkin = df_checkin[(df_checkin.long * -0.77413 - 16.3008 > df_checkin.lat)
# 								& (df_checkin.long * 3.290917 + 283.292 < df_checkin.lat)
# 								& (df_checkin.long * 0.162446 + 52.56053 < df_checkin.lat)
# 								& (df_checkin.long * 1.206121 + 130.0629 > df_checkin.lat)]

# print(filtereddf_checkin.sort_values(by=['long', 'lat']))
filtereddf_checkin['time_str'] = filtereddf_checkin['time'].astype(str).str[:10]

grouped_df2 = filtereddf_checkin.groupby(['loc_id']).user_id.nunique()
active_loc = grouped_df2[grouped_df2 >= 3].index.tolist()

grouped_df = filtereddf_checkin.groupby(['user_id']).time_str.nunique()
active_user = grouped_df[grouped_df >= 5].index.tolist()

# print(grouped_df)
# print(active_user)


SF_checkin = filtereddf_checkin[filtereddf_checkin.user_id.isin(active_user)]
SF_checkin = SF_checkin[SF_checkin.loc_id.isin(active_loc)]

grouped_df2 = SF_checkin.groupby(['loc_id']).user_id.nunique()
active_loc = grouped_df2[grouped_df2 >= 3].index.tolist()

grouped_df = SF_checkin.groupby(['user_id']).time_str.nunique()
active_user = grouped_df[grouped_df >= 5].index.tolist()

SF_checkin = SF_checkin[SF_checkin.user_id.isin(active_user)]
SF_checkin = SF_checkin[SF_checkin.loc_id.isin(active_loc)]

loc_df = pd.DataFrame(SF_checkin['loc_id'].unique())
loc_df['index1'] = loc_df.index
loc_df.columns = ['loc_id', 'index1']
print(len(loc_df))

df = pd.merge(SF_checkin, loc_df, how='inner', left_on=['loc_id'], right_on=['loc_id']) \
	[['user_id', 'index1', 'time']]
df.columns = ['user_id', 'loc_id', 'time']
df = df.sort_values(by=['user_id', 'time'])
print(len(df['user_id'].unique()))

# print(df)
# print(len(active_user))
# print(len(active_loc))

# df.to_csv("C:/Users/ducnm/Downloads/NYcheckin_0606.csv")
