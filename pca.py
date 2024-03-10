from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

data = pd.read_csv('../data/preprocess/processed.csv')
data['elapsed_time'] = pd.to_timedelta(data['elapsed_time'])
data['elapsed_time'] = data.groupby('match_id')['elapsed_time'].transform(lambda x: x - x.min())
data['elapsed_time'] = data['elapsed_time'].dt.total_seconds() / 60

data_player1 = data[['elapsed_time', 'p1_sets', 'p1_games', 'p1_score', 'server', 'point_victor', 
                     'game_victor','p1_points_won', 'set_victor', 'p1_ace', 'p1_winner', 'p1_double_fault', 
                     'p1_break_pt_won', 'p1_unf_err', 'p1_net_pt', 'p1_net_pt_won', 'p1_break_pt', 'p1_break_pt_won', 
                     'p1_break_pt_missed', 'p1_distance_run', 'p1_repeat_win', 'serve_depth', 'return_depth']]
data_player2 = data[['elapsed_time', 'p2_sets', 'p2_games', 'p2_score', 'server', 'point_victor', 
                     'game_victor','p2_points_won', 'set_victor', 'p2_ace', 'p2_winner', 'p2_double_fault', 
                     'p2_break_pt_won', 'p2_unf_err', 'p2_net_pt', 'p2_net_pt_won', 'p2_break_pt', 'p2_break_pt_won', 
                     'p2_break_pt_missed', 'p2_distance_run', 'p2_repeat_win', 'serve_depth', 'return_depth']]

# 填充缺失值
imputer = SimpleImputer(strategy='mean')
data_player1_imputed = imputer.fit_transform(data_player1)
data_player2_imputed = imputer.fit_transform(data_player2)

# 标准化数据
scaler = StandardScaler()
data_player1_scaled = scaler.fit_transform(data_player1_imputed)
data_player2_scaled = scaler.fit_transform(data_player2_imputed)

# 创建PCA对象
pca = PCA(n_components=1)

# 使用PCA对象对数据进行转换
data_player1_pca = pca.fit_transform(data_player1_scaled)
print("data_player1的主成分信息量：", pca.explained_variance_ratio_)
data_player2_pca = pca.fit_transform(data_player2_scaled)
print("data_player2的主成分构成：", pca.components_)
print("data_player2的主成分信息量：", pca.explained_variance_ratio_)

# 将结果保存为新的DataFrame
data_player1_pca_df = pd.DataFrame(data=data_player1_pca, columns=['Principal Component 1'])
data_player2_pca_df = pd.DataFrame(data=data_player2_pca, columns=['Principal Component 1'])
data_merged = pd.concat([data_player1_pca_df, data_player2_pca_df], axis=1)

# data_merged.to_csv('../data/preprocess/pca.csv', index=False)