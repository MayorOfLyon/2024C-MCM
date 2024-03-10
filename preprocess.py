import pandas as pd
import numpy as np

# deep learning
data = pd.read_csv('../data/preprocess/tennis.csv')
data_dl = data[['OT', 'momentum2']]
len = len(data_dl)
time_range = pd.date_range(start='2023-06-01', periods=len, freq='1min')
time_range_df = pd.DataFrame(time_range, columns=['date'])

data_dl = pd.concat([time_range_df, data_dl], axis=1)
data_dl.to_csv('../data/preprocess/data_dl.csv', index=False)
