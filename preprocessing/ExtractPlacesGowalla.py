import pandas as pd

inputFile = "D:\Research\Dataset\Gowalla_totalCheckins.txt"
data = pd.read_csv(inputFile, sep='\t', header=None)
print(len(data))

# print(data)
x1 = data.groupby([4, 2, 3])[0]
# x1_list = x1.apply(list)
# print(x1_list)

# x1_count = x1.agg('count')
# print(x1_count.sort_values(0, ascending=False))
# x1_count.to_csv('D:/Research/Dataset/checkin/Gowalla_places.txt')
# for count in x1_count:
#     print(count)

