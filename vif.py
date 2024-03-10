from scipy.stats import kendalltau
import pandas as pd

data = pd.read_csv('../data/preprocess/momentum.csv')
x = data['momentum1']
y = data['momentum2']

# 计算Kendall Rank
kendall_corr, p_value = kendalltau(x, y)

# 打印结果
print(f"Kendall Rank correlation coefficient: {kendall_corr}")
print(f"P-value: {p_value}")
