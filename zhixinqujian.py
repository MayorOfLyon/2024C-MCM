import numpy as np
import matplotlib.pyplot as plt

# 读取真实数据和预测数据
true_data = np.array([[ 6.9148608e-01,  7.1566872e-01],
                      [ 5.7990301e-01,  6.8765748e-01],
                      [ 5.5471677e-01,  7.9122353e-01],
                      [ 5.7930768e-01,  9.0655237e-01],
                      [ 6.0339713e-01,  0.8944044e+00],
                      [ 6.3255531e-01,  1.0448939e+00],
                      [ 5.1216173e-01,  1.1866648e+00],
                      [ 6.6305959e-01,  1.2611080e+00],
                      [ 6.7248917e-01,  1.1463671e+00],
                      [ 9.1835321e-01,  0.9589154e+00],
                      [ 9.9117267e-01,  1.1375423e+00],
                      [ 8.9437735e-01,  0.9291474e+00]])

pred_data = np.array([[ 7.41635263e-01,  7.33037710e-01],
                      [ 5.67809403e-01,  7.17530787e-01],
                      [ 5.75067937e-01,  7.29412735e-01],
                      [ 5.98839283e-01,  8.36448014e-01],
                      [ 6.10182226e-01,  7.83502698e-01],
                      [ 7.76638687e-01,  1.00448680e+00],
                      [ 6.69323623e-01,  1.12233668e+00],
                      [ 7.87061632e-01,  1.08076281e+00],
                      [ 6.05823696e-01,  9.27357769e-01],
                      [ 8.23359387e-01,  1.03597665e+00],
                      [ 8.73827932e-01,  1.10112560e+00],
                      [ 6.86050675e-01,  1.00616360e+00]])

# 提取 p1 和 p2 的数据
true_p1 = true_data[:, 0]
true_p2 = true_data[:, 1]

pred_p1 = pred_data[:, 0]
pred_p2 = pred_data[:, 1]

# 创建置信区间范围
# 这里简单地假设置信区间为预测值的标准差
pred_mean1 = np.mean(pred_p1)
pred_std1 = np.std(pred_p1)

# 计算置信区间
ci_lower1 = pred_mean1 - 1.96 * pred_std1
ci_upper1 = pred_mean1 + 1.96 * pred_std1

pred_mean2 = np.mean(pred_p2)
pred_std2 = np.std(pred_p2)

ci_lower2 = pred_mean2 - 1.96 * pred_std2
ci_upper2 = pred_mean2 + 1.96 * pred_std2

# 画出折线图
plt.figure(figsize=(10, 6))
plt.plot(true_p1, label='p1_True')
plt.plot(true_p2, label='p2_True')
plt.plot(pred_p1, label='p1_Predicted', linestyle='dashed')
plt.plot(pred_p2, label='p2_Predicted', linestyle='dashed')

# 画出置信区间范围
plt.fill_between(range(len(pred_p1)), true_p1 - ci_lower1, true_p1 + ci_upper1, color='#82f5f1', alpha=0.3)
plt.fill_between(range(len(pred_p2)), true_p2 - ci_lower2, true_p2 + ci_upper2, color='orange', alpha=0.3)

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.title('Predicted vs True Values with Confidence Interval')
plt.ylabel('P')
plt.legend(loc='upper left',fontsize='large')
plt.grid(True)
plt.show()
