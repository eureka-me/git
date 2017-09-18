#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/17
# Filename: p69 

__author__ = 'takutohasegawa'
__date__ = "2017/09/17"

import pymc as pm
import numpy as np
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize


np.set_printoptions(precision=3, suppress=True)

challenger_data = np.genfromtxt("data/challenger_data.csv",
                                skip_header=1, usecols=[1, 2],
                                missing_values="NA",
                                delimiter=",")

# NaNを削除
challenger_data = challenger_data[~np.isnan(challenger_data[:, 1])]

# 気温
temperature = challenger_data[:, 0]

# 破損しているかどうか
D = challenger_data[:, 1]

beta = pm.Normal("beta", 0, 0.001, value=0)
alpha = pm.Normal("alpha", 0, 0.001, value=0)

def logistic(x, beta, alpha=0):
    return 1. / (1. + np.exp(np.dot(beta, x) + alpha))
    # np.dotは引数の配列の内積を求める

@pm.deterministic
def p(t=temperature, alpha=alpha, beta=beta):
    """
    p(t)はロジスティック関数
    :param t: 観測した外気温
    :param alpha: パラメータ
    :param beta: パラメータ
    :return: 破損する確率
    """
    return 1. / (1. + np.exp(beta * t + alpha))

observed = pm.Bernoulli("bernoulli_obs", p, value=D, observed=True)
model = pm.Model([observed, beta, alpha])

map_ = pm.MAP(model)
map_.fit()
mcmc = pm.MCMC(model)
mcmc.sample(120000, 100000, 2)
# mcmc.sample(1200, 1000, 2)


alpha_samples = mcmc.trace('alpha')[:, None]
beta_samples = mcmc.trace('beta')[:, None]

"""
# P.71のグラフ
figsize(12.5, 6)

plt.subplot(211)
plt.hist(beta_samples, bins=35, color="red", normed=True, label=r"posterior of beta")
plt.legend()

plt.subplot(212)
plt.hist(alpha_samples, bins=35, color="blue", normed=True, label=r"posterior of alpha")
plt.legend()

plt.show()

"""

t = np.linspace(temperature.min()-5, temperature.max()+5, 50)[:, None]
p_t = logistic(t.T, beta_samples, alpha_samples)
mean_prob_t = p_t.mean(axis=0)


"""
# P.72のグラフ
figsize(12.5, 4)

plt.plot(t, mean_prob_t, lw=3, label="average posterior probability of defect")
plt.plot(t, p_t[0, :], ls="--", label="realization from posterior")
plt.plot(t, p_t[-2, :], ls="--", label="realization from posterior")

plt.scatter(temperature, D, color="k", s=50, alpha=0.5)
plt.legend()
plt.ylim(-0.1, 1.1)
plt.xlim(t.min(), t.max())
plt.show()
"""

"""
# P.73のグラフ
from scipy.stats.mstats import mquantiles
figsize(12.5, 4)

# 信頼区間の上下2.5%
qs = mquantiles(p_t, [0.025, 0.975], axis=0)
plt.fill_between(t[:, 0], *qs, alpha=0.7, color="gray")
plt.plot(t[:, 0], qs[0], label="95% CI", color="red", alpha=0.7)
plt.plot(t, mean_prob_t, lw=1, ls="--", color="k", label="average posterior probability of defect")
plt.xlim(t.min(), t.max())
plt.ylim(-0.02, 1.02)
plt.legend(loc="lower left")

plt.scatter(temperature, D, color="k", s=50, alpha=0.5)
plt.show()
"""

# P.74のグラフ
figsize(12.5, 2.5)

prob_31 = logistic(31, beta_samples, alpha_samples)
plt.hist(prob_31, bins=1000, normed=True)
plt.xlim(.995, 1)
plt.show()

