import pandas as pd
import numpy as np

last_fm = pd.read_csv("C:/Users/ducnm/Downloads/userid_artistid_time1.csv", sep=',', header=0, index_col='Unnamed: 0')
print(len(last_fm))
print(last_fm.groupby(['userid'])['artistid'].count())
# print(last_fm[["userid", "artistid", "time"]])
