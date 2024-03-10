import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import json

# 读取数据
data = pd.read_csv('../data/preprocess/momentum_ew.csv')
data_momentum = data[['match_id', 'player1', 'player2', 'momentum1', 'momentum2']]

# 从data_momentum中找到player1或player2为Carlos Alcaraz所有行
data_carlos = data_momentum[(data_momentum['player1'] == 'Carlos Alcaraz') | (data_momentum['player2'] == 'Carlos Alcaraz')]

# 找到player2为Carlos Alcaraz的行
data_carlos1 = data_momentum[data_momentum['player1'] == 'Carlos Alcaraz']
data_carlos2 = data_momentum[data_momentum['player2'] == 'Carlos Alcaraz']

# 将player2为Carlos Alcaraz的行的player1和player2交换，momentum1和momentum2交换
data_carlos2 = data_carlos2.rename(columns={'player1': 'player2', 'player2': 'player1', 'momentum1': 'momentum2', 'momentum2': 'momentum1'})

# 将data_carlos1与data_carlos2按照列名合并
data_carlos = pd.concat([data_carlos1, data_carlos2])
data_carlos = data_carlos[['match_id', 'momentum1']]

# 将data_carlos按照match_id拆分为若干dataframe
data_carlos = data_carlos.groupby('match_id')
data_carlos = [group for _, group in data_carlos]

# 创建一个空的 DataFrame 来存储结果
result = pd.DataFrame(index=range(max(len(df) for df in data_carlos)))

# 遍历 data_carlos 列表中的每个 DataFrame
for i, df in enumerate(data_carlos):
    # 将 'momentum1' 列添加到结果 DataFrame 中
    result['momentum1_' + str(i)] = pd.Series(df['momentum1'].values)

result = result.dropna(axis = 0)

# 求平均
momentum_carlos = pd.DataFrame()
momentum_carlos['momentum'] = result.mean(axis=1)

# 找到数据中和Novak Djokovic比赛的选手
data_novak = data_momentum[(data_momentum['player1'] == 'Novak Djokovic') | (data_momentum['player2'] == 'Novak Djokovic')]
# 删除和Carlos Alcaraz比赛的数据
data_novak = data_novak[~data_novak['match_id'].isin(data_carlos1['match_id'])]
# 找到player2为Novak Djokovic的行
data_novak1 = data_momentum[data_momentum['player1'] == 'Novak Djokovic']
data_novak2 = data_momentum[data_momentum['player2'] == 'Novak Djokovic']
# 交换
data_novak2 = data_novak.rename(columns={'player1': 'player2', 'player2': 'player1', 'momentum1': 'momentum2', 'momentum2': 'momentum1'})
# merge
data_novak = pd.concat([data_novak1, data_novak2])
data_novak = data_novak[['match_id','player2', 'momentum2']]

data_novak = data_novak.groupby('match_id')
data_novak = [group for _, group in data_novak]
similarity = []
for i, df in enumerate(data_novak):
    # 计算余弦相似度
    # print(data_novak[i]['player2'])
    vec1 = data_novak[i]['momentum2'].values.reshape(1, -1)
    vec2 = momentum_carlos.values.reshape(1, -1)

    # 确保 vec1 和 vec2 的长度相同
    if vec1.shape[1] > vec2.shape[1]:
        vec1 = vec1[:,:vec2.shape[1]]
    elif vec1.shape[1] < vec2.shape[1]:
        vec2 = vec2[:,:vec1.shape[1]]

    # 现在你可以计算它们的余弦相似度
    cos_sim = cosine_similarity(vec1, vec2)
    similarity.append(cos_sim)
    # print(cos_sim)
    
# 找到match_id为1602的行
data_search = data[data['match_id'].isin(['2023-wimbledon-1602', '2023-wimbledon-1408', '2023-wimbledon-1601', '2023-wimbledon-1501'])]
data_search['cha'] = data_search['momentum1'] - data_search['momentum2']

# 对cha均匀分为三个桶
data_search['cha'] = pd.qcut(data_search['cha'], 3, labels=['low', 'medium', 'high'])

# 创建新的一列，若p1_sets_won > p2_sets_won，则为1，若p1_sets_won < p2_sets_won，则为-1，否则为0
data_search['set_win'] = np.where(data_search['p1_sets'] > data_search['p2_sets'], 1, np.where(data_search['p1_sets'] < data_search['p2_sets'], -1, 0))

data_search['cha_set_win'] = data_search.apply(lambda row: f"{row['cha']}_{row['set_win']}", axis=1)
data_search['cha_set_win_next'] = data_search['cha_set_win'].shift(-1)

data_search = data_search.rename(columns={'final_metric': 'action', 'cha_set_win': 'state', 'cha_set_win_next': 'next_state'})
order = ['low_-1', 'medium_-1', 'high_-1', 'low_0', 'medium_0', 'high_0', 'low_1', 'medium_1', 'high_1']
data_search['state'] = pd.Categorical(data_search['state'], categories=order, ordered=True).codes
data_search['next_state'] = pd.Categorical(data_search['next_state'], categories=order, ordered=True).codes
data_search = data_search[data_search['next_state'] != -1]

counts1 = data_search.groupby(['state']).size()
# print(counts1)

counts2 = data_search.groupby(['state', 'action']).size()
# print(counts2)


counts3 = data_search.groupby(['state', 'action', 'next_state']).size()
# print(counts3)

# 统计每个state下出action的概率
probs = counts2 / counts1
probs1 = probs.unstack()
print(probs)
probs_matrix_np = np.array(probs1)
# print(probs_matrix_np)
# heatmap
fig, ax = plt.subplots()
cax = ax.matshow(probs_matrix_np, cmap='hot')
fig.colorbar(cax)
plt.show()
np.savetxt('probs_matrix.txt', probs_matrix_np, fmt='%f')

# 统计不同state下出不同action后，下一个state的概率
probs = counts3 / counts2
probs2 = probs.unstack()
probs2 = probs2.fillna(0)
print(probs2)
probs_matrix_np = np.array(probs2)
probs_matrix_np = probs_matrix_np.reshape(9, 2, 9)

probs_matrix_np = np.around(probs_matrix_np, 2)
row_sums = probs_matrix_np.sum(axis=2, keepdims=True)
probs_matrix_np = probs_matrix_np / row_sums
# print(probs_matrix_np)
np.save('probs_matrix_next.txt', probs_matrix_np)

row_sums = probs_matrix_np.sum(axis=2)

# 找出和不为1的行
not_one_rows = np.where(np.abs(row_sums - 1) > 1e-6)

# 打印结果
# print(not_one_rows)