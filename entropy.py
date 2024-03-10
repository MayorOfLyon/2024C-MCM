import pandas as pd
import numpy as np

def vec2log(vector):
    ans = []
    for i in range(len(vector)):
        if vector[i] == 0:
            ans.append(0)
        else:
            ans.append(np.log(vector[i]))
    return np.array(ans)

data = pd.read_csv('../data/original_data/Wimbledon_featured_matches.csv')
data = data[['server', 'p1_ace', 'p1_winner', 'p1_double_fault', 'p1_unf_err', 'p1_net_pt', 'p1_net_pt_won',
             'p1_break_pt', 'p1_break_pt_won', 'p1_break_pt_missed', 'p1_distance_run', 'rally_count']]
print(data.shape)
# 熵权法
# 1.正向化
data_rest = data[['p1_ace', 'p1_winner', 'p1_net_pt', 'p1_net_pt_won',
                  'p1_break_pt', 'p1_break_pt_won', 'p1_distance_run', 'rally_count']]
data_zhengxiang = data[['p1_double_fault', 'p1_unf_err','p1_break_pt_missed',]]
data_zhengxiang = data_zhengxiang.max() - data_zhengxiang

data = pd.concat([data_rest, data_zhengxiang], axis=1)

# # 检查是否有负数
# is_negative = data.lt(0).any()
# print(is_negative)

# # 检查缺失值
# print("sbbb ",data.isnull().sum())

# 2. 标准化
for index, column in enumerate(data.columns):
    data[column] = data[column] / np.sqrt((data[column] ** 2).sum())

# 2.计算每个指标的熵值
# 熵权法
features = len(data.columns)
weights = np.zeros(features)
entropys = np.zeros(features)
for index, column in enumerate(data.columns):
    print(column)
    vector = data[column].values
    norm_vector = vector / vector.sum()
    # 信息熵
    entropy = - np.sum((norm_vector * vec2log(norm_vector))) / np.log(len(data))
    weights[index] = 1 - entropy
    entropys[index] = entropy
weights = weights / weights.sum()
print("熵权法权重：", weights)
print("熵权法熵值：", entropys)

# 加权
data['final_metric'] = data[data.columns].dot(weights)

# 对final_metric分成两个桶
new_data = pd.read_csv('../data/preprocess/momentum.csv')
data['final_metric'] = pd.qcut(data['final_metric'], 2, labels=[0, 1])
data = pd.concat([new_data, data['final_metric']], axis=1)
data.to_csv('../data/preprocess/momentum_ew.csv', index=False)