import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('../data/preprocess/momentum.csv')
data = data[['match_id', 'momentum1', 'momentum2', 'p1_points_won', 'p2_points_won']]

# 获取所有的 match_id
matches = data['match_id'].unique()

last_values = []
areas1 = []
# 对每个 match_id 执行计算
for match in matches:
    match_data = data[data['match_id'] == match]
    
    points_win = match_data['p1_points_won'] / (match_data['p1_points_won'] + match_data['p2_points_won'])
    last_value = points_win.iloc[-1]

    # plot
    x = np.arange(len(match_data))
    y1 = match_data['momentum1']
    cs1 = CubicSpline(x, y1)
    x_interp = np.linspace(min(x), max(x), 5000)
    y_interp1 = cs1(x_interp)

    y2 = match_data['momentum2']
    cs2 = CubicSpline(x, y2)
    y_interp2 = cs2(x_interp)

    # plt.plot(x_interp, y_interp1, label='momentum1')
    # plt.plot(x_interp, y_interp2, label='momentum2')
    # plt.legend()
    # plt.show()

    # momentum1围绕坐标轴的面积
    area1 = np.trapz(y_interp1, dx=x_interp[1]-x_interp[0])
    area2 = np.trapz(y_interp2, dx=x_interp[1]-x_interp[0])
    total_area = area1 + area2
    
    # print('match_id:', match)
    # print('area1:', area1/total_area)
    # print('area2:', area2/total_area)
    # print('last_value:', last_value)
    # print('------------------------')
    last_values.append(last_value)
    areas1.append(area1/total_area)
# 计算last_values与area1的相关系数
print('correlation:', np.corrcoef(last_values, areas1)[0, 1])
# spearman
print('spearman:', pd.Series(last_values).corr(pd.Series(areas1), method='spearman'))