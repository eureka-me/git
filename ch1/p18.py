#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/12
# Filename: p18 

__author__ = 'takutohasegawa'
__date__ = "2017/09/12"

import pymc as pm
# pymc3はなぜか動かない！！のでインタプリタをanaconda2にする必要あり。

from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt

count_data = np.loadtxt("./data/txtdata.csv")
n_count_data = len(count_data)

alpha = 1. / count_data.mean()

"""stochastic変数の定義"""
lambda_1 = pm.Exponential("lambda_1", alpha)
lambda_2 = pm.Exponential("lambda_2", alpha)
tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data)

# stochastic変数はバックエンドで乱数発生器のように振る舞う
# random()メソッドでstocastic変数から分布に従った乱数を発生させる
print(lambda_1.random(), tau.random())

@pm.deterministic
def lambda_(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
    # tau, lambda_1, lambda_2は引数としているが、デフォルトで設定されるようになっている
    out = np.zeros(n_count_data)
    out[:tau] = lambda_1
    out[tau:] = lambda_2
    return out

"""ポワソン分布オブジェクトの定義"""
observation = pm.Poisson('obs', lambda_, value=count_data, observed=True)
# パラメータにlambda_、データcount_dataをvalueキーワードで受け渡し、それを変数observationが受け取る
# さらにobserved=Trueとして、この値は観測地なので解析時には固定されているということをPymcに伝えている

"""モデルインスタンスの生成"""
model = pm.Model([observation, lambda_1, lambda_2, tau])

"""Markov chain Monte Carlo; MCMC オブジェクトの生成"""
mcmc = pm.MCMC(model)
# mcmc.sample(40000, 10000)
mcmc.sample(400, 100)

lambda_1_samples = mcmc.trace('lambda_1')[:]
lambda_2_samples = mcmc.trace('lambda_2')[:]
tau_samples = mcmc.trace('tau')[:]

"""ヒストグラムの描画"""

"""
ax = plt.subplot(311)
ax.set_autoscaley_on(False)
plt.hist(lambda_1_samples, histtype='stepfilled',
         bins=30, alpha=0.85, color="red", normed=True,
         label="posterior of lambda_1")
plt.legend(loc="upper left")
plt.title("Posterior distributions of the pearameters "
          r"lambda_1, lambda_2, tau")
plt.xlim([15, 30])
plt.xlabel("lambda_1 value")
plt.ylabel("Density")

ax = plt.subplot(312)
plt.hist(lambda_2_samples, histtype='stepfilled',
         bins=30, alpha=0.85, color="green", normed=True,
         label="posterior of lambda_2")
plt.legend(loc="upper left")
plt.xlim([15,30])
plt.xlabel("lambda_2 value")
plt.ylabel("Density")

plt.subplot(313)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins=n_count_data, alpha=1,
         label=r"posterior of tau", color="blue",
         weights=w, rwidth=2.)
plt.xticks(np.arange(n_count_data))
plt.legend(loc="upper left")
plt.ylim([0,.75])
plt.xlim([35, len(count_data) -20])
plt.xlabel("tau (in days)")
plt.ylabel("Probability")
"""

N = tau_samples.shape[0]
expected_text_per_day = np.zeros(n_count_data) # データ数

for day in range(0, n_count_data):
    # ixは、dayの値よりも前の変化点に対応するtauの
    # すべてのサンプルのboolインデックス
    ix = day < tau_samples

    # 事後分布からの各サンプルはtauの値に対応している。それぞれの日において
    # tauの値は変化点よりも「前（lambda_1の期間）」か「後（lambda_2の期間）」
    # かを表している。lambda_1とlambda_2からそれぞれサンプリングして、全ての
    # サンプルについての平均をとれば、その日におけるlambdaの期待値が得られる。
    # 説明しているように、「メッセージ数」の確立変数はポワソン分布に従うので、
    # lambdaはメッセージ数の期待値である。

    expected_text_per_day[day] = (lambda_1_samples[ix].sum() +
                                  lambda_2_samples[~ix].sum()) / N

plt.plot(range(n_count_data), expected_text_per_day,
         lw=4, color="blue", label="Expected number of "
                                   "text messages received")
plt.xlim(0, n_count_data)
plt.ylim(0,60)
plt.xlabel("Day")
plt.ylabel("Number of text messages")
plt.title("Number of text messages received versus "
          "expected number received")

plt.bar(np.arange(len(count_data)), count_data, color="red", alpha=0.65,
        label="observed text messages par day")
plt.legend(loc="upper left")

plt.show()

