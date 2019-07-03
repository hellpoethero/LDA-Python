import pandas as pd
import numpy as np

last_fm = pd.read_csv("C:/userid-timestamp-artid-artname-traid-traname.tsv", sep='\t', header=None, error_bad_lines=False)
print(len(last_fm))

# tracklist = list(set(last_fm[5].tolist()))
# print(len(tracklist))

last_fm = last_fm.groupby(2).filter(lambda x: len(x) >= 100)
print(len(last_fm))

userlist = list(set(last_fm[0].tolist()))
print(len(userlist))
user_df = pd.DataFrame(userlist)
user_df['userid'] = user_df.index
artistlist = list(set(last_fm[2].tolist()))
print(len(artistlist))
artist_df = pd.DataFrame(artistlist)
artist_df['artistid'] = artist_df.index

tracklist = list(set(last_fm[5].tolist()))
print(len(tracklist))

# print(user_df)
# print(artist_df)

# a = pd.merge(last_fm, user_df, how="inner", on=[0, 0])
# b = pd.merge(a, artist_df, how="inner", left_on=2, right_on=0)
# c = b[["userid", "artistid", 1]]
# c = c.rename(index=str, columns={1: "time"})
# c = c.sort_values(by=['userid', 'time'])
# c = c[['userid', 'artistid', 'time']].reset_index()
# print(c)
# c.to_csv("C:/Users/ducnm/Downloads/userid_artistid_time1.csv")

# last_fm1 = last_fm[[0, 1, 2]]
# print(last_fm1)

# g = last_fm.groupby(2)[0].count().sort_values(0, ascending=True)
# print(g)

# read => extract data with same form as gowalla