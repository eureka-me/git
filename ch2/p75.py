#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/17
# Filename: p75 

__author__ = 'takutohasegawa'
__date__ = "2017/09/17"

import pymc as pm
import numpy as np
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize


# np.set_printoptions(precision=3, suppress=True)
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

simulated = pm.Bernoulli("bernoulli_sim", p)
N = 10000

mcmc = pm.MCMC([simulated, alpha, beta, observed])
mcmc.sample(N)

simulations = mcmc.trace("bernoulli_sim")[:].astype(int)

"""
# P.76のグラフ
figsize(12.5, 10)
for i in range(4):
    ax = plt.subplot(4, 1, i+1)
    plt.scatter(temperature, simulations[1000 * i, :],
                color="k", s=50, alpha=0.6)
plt.show()
"""

posterior_probability = simulations.mean(axis=0)

# 観測、破損発生の有無を生成したデータ、破損発生の事後確率、破損発生
print("Obs. | Array of Simulated Defects | Posterior   | Realized")
print("                                    Probability   Defect  ")
print("                                    of Defect             ")

for i in range(len(D)):
    print("%-4s | %s | %-12.2f | %d" %
          (str(i).zfill(2),
           str(simulations)[:-1] + "...]".ljust(6),
           posterior_probability[i], D[i]))


