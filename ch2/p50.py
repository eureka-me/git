#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/16
# Filename: p50 

__author__ = 'takutohasegawa'
__date__ = "2017/09/16"

import pymc as pm
from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt

figsize(12,4)
true_p_A = 0.05
true_p_B = 0.049

# サンプルサイズがかなり違うが、ベイズ解析では問題ない
N_A = 15000
N_B = 7500

# 観測データを生成
observations_A = pm.rbernoulli(true_p_A, N_A)
observations_B = pm.rbernoulli(true_p_B, N_B)

# 最初の30個だけ表示
print("Obs from Site A: ", observations_A[:30].astype(int), "...")
print("Obs from Site B: ", observations_B[:30].astype(int), "...")

print("\n")
print("A mean: ", observations_A.mean())
print("B mean: ", observations_B.mean())

"""PyMCモデルの設定"""
# ここでもp_Aとp_Bの事前分布は一様分布と仮定する
p_A = pm.Uniform("p_A", 0, 1)
p_B = pm.Uniform("p_B", 0, 1)

# deterministic変数のdeltaの宣言。これも推論する

@pm.deterministic
def delta(p_A=p_A, p_B=p_B):
    return p_A - p_B

# 観測データの設定：今回の観測データセットは二つ
obs_A = pm.Bernoulli("obs_A", p_A, value=observations_A, observed=True)
obs_B = pm.Bernoulli("obs_B", p_B, value=observations_B, observed=True)

mcmc = pm.MCMC([p_A, p_B, delta, obs_A, obs_B])
mcmc.sample(25000, 5000)

"""描画"""
p_A_samples = mcmc.trace("p_A")[:]
p_B_samples = mcmc.trace("p_B")[:]
delta_samples = mcmc.trace("delta")[:]

figsize(12.5, 10)
ax = plt.subplot(311)
plt.hist(p_A_samples, bins=30, alpha=0.5 ,label="posterior of p_A", color="red", normed=True)
plt.vlines(true_p_A, 0, 80, linestyles="--", label="true p_A")
plt.legend(loc="upper right")
plt.xlim(0, .1)
plt.ylim(0, 80)

ax = plt.subplot(312)
plt.hist(p_B_samples, bins=30, alpha=0.5 ,label="posterior of p_B", color="red", normed=True)
plt.vlines(true_p_B, 0, 80, linestyles="--", label="true p_B")
plt.legend(loc="upper right")
plt.xlim(0, .1)
plt.ylim(0, 80)

ax = plt.subplot(313)
plt.hist(delta_samples, bins=30, alpha=0.5 ,label="posterior of delta", color="red", normed=True)
plt.vlines(true_p_A - true_p_B, 0, 60, linestyles="--", label="true delta")
plt.legend(loc="upper right")
# plt.xlim(0, .1)
# plt.ylim(0, 80)


print("Probability site A is WORSE than site B: %.3f" % (delta_samples < 0).mean())
print("Probability site A is BETTER than site B: %.3f" % (delta_samples > 0).mean())


plt.show()

