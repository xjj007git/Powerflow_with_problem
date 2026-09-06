#这里是将一维数组转化成二维数组，并且存入excel表格中
import math
import pandas as pd
data = list(range(288))
cols = 18
rows = math.ceil(len(data) / cols)
data = data + [0] * (rows * cols - len(data))
data_2d = pd.DataFrame(data).values.reshape(rows, cols)
df = pd.DataFrame(data_2d)
df.to_excel("./data/output.xlsx", index=False, header=False)