import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import johnsonsu
import seaborn as sns
from scipy.stats import lognorm
from scipy.stats import norm
from scipy import stats

# 读取数据
data = pd.read_csv('../data/preprocess/momentum.csv')

momentum = pd.concat([data['momentum1'], data['momentum2']], axis=0)

# mu, std = np.mean(momentum), np.std(momentum)
# d, p = stats.kstest((momentum - mu) / std, 'norm')
# print("p = {:g}".format(p))
# alpha = 0.05
# if p < alpha:  # null hypothesis: x comes from a normal distribution
#     print("The null hypothesis can be rejected")
# else:
#     print("The null hypothesis cannot be rejected")

# skewness = stats.skew(momentum)
# kurtosis = stats.kurtosis(momentum)
# print("Skewness: ", skewness)
# print("Kurtosis: ", kurtosis)

# k2, p = stats.normaltest(momentum)
# print("p = {:g}".format(p))
# alpha = 0.05
# if p < alpha:  # null hypothesis: x comes from a normal distribution
#     print("The null hypothesis can be rejected")
# else:
#     print("The null hypothesis cannot be rejected")
    
    
    
# params = johnsonsu.fit(momentum.dropna())
# x = np.linspace(min(momentum.dropna()), max(momentum.dropna()), 100)
# pdf_fitted = johnsonsu.pdf(x, params[0], params[1], loc=params[2], scale=params[3])
# sns.histplot(momentum.dropna(), kde=True, stat='density')
# plt.plot(x, pdf_fitted, 'r-')
# plt.title('Johnson SU fit for momentum')
# plt.show()
# print(params)

# a, b, loc, scale = stats.johnsonsu.fit(momentum)
# theoretical = stats.johnsonsu.rvs(a, b, loc, scale, size=len(momentum))
# ks_stat, ks_p_value = stats.ks_2samp(momentum, theoretical)
# print(a,b,loc,scale)
# print('KS Statistic:', ks_stat)
# print('KS P-Value:', ks_p_value)

momentum = momentum + 5.8
momentum = np.log(momentum)
params = norm.fit(momentum.dropna())
print(params)


momentum = momentum + 5.8
params = lognorm.fit(momentum.dropna())
x = np.linspace(min(momentum.dropna()), max(momentum.dropna()), 100)
pdf_fitted = lognorm.pdf(x, params[0], loc=params[1], scale=params[2])
sns.histplot(momentum.dropna(), kde=True, stat='density')
plt.plot(x, pdf_fitted, 'r-')
plt.title('Log-normal fit for momentum')
plt.show()
print(params)


# log_momentum = np.log(momentum)
# k2, p = stats.normaltest(log_momentum)
# print("p = {:g}".format(p))
# alpha = 0.05
# if p < alpha:  # null hypothesis: x comes from a normal distribution
#     print("The null hypothesis can be rejected")
# else:
#     print("The null hypothesis cannot be rejected")
    
# params = norm.fit(momentum.dropna())
# x = np.linspace(min(momentum.dropna()), max(momentum.dropna()), 100)
# pdf_fitted = norm.pdf(x, loc=params[0], scale=params[1])
# sns.histplot(momentum.dropna(), kde=True, stat='density')
# plt.plot(x, pdf_fitted, 'r-')
# plt.title('Normal fit for momentum')
# plt.show()
