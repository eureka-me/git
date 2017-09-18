#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/17
# Filename: p88 

__author__ = 'takutohasegawa'
__date__ = "2017/09/17"

import pymc as pm
import numpy as np
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D


# 観測データの生成

# 観測データのサンプルサイズ
N = 1

# 真のパラメータ
lambda_1_true = 1
lambda_2_true = 3

# そして二つのパラメータからデータを生成する
data = np.concatenate([
    stats.poisson.rvs(lambda_1_true, size=(N, 1)),
    stats.poisson.rvs(lambda_2_true, size=(N, 1)),
], axis=1)
print("observed (2-dimensional, sample size = %d:" % N, data)


# プロット
x = y = np.linspace(.01, 5, 100)
likelihood_x = np.array([stats.poisson.pmf(data[:, 0], _x) for _x in x]).prod(axis=1)
likelihood_y = np.array([stats.poisson.pmf(data[:, 0], _y) for _y in y]).prod(axis=1)
L = np.dot(likelihood_x[:, None], likelihood_y[None, :])

jet = plt.cm.jet

figsize(12.5, 12)

plt.subplot(221)
uni_x = stats.uniform.pdf(x, loc=0, scale=5)
uni_y = stats.uniform.pdf(x, loc=0, scale=5)
M = np.dot(uni_x[:, None], uni_y[None, :])
plt.imshow(M, interpolation='none', origin='lower', cmap=jet, vmax=1, vmin=-.15, extent=(0, 5, 0, 5))
plt.scatter(lambda_2_true, lambda_1_true, c='k', s=50, edgecolor='none')
plt.xlim(0, 5)
plt.ylim(0, 5)

plt.subplot(223)
plt.contour(x, y, M * L) # 尤度と事前分布の積が事後分布になるとのこと
plt.imshow(M * L, interpolation='none', origin='lower', cmap=jet, extent=(0, 5, 0, 5))
plt.scatter(lambda_2_true, lambda_1_true, c='k', s=50, edgecolor='none')
plt.xlim(0, 5)
plt.ylim(0, 5)

plt.subplot(222)
exp_x = stats.expon.pdf(x, loc=0, scale=3)
exp_y = stats.expon.pdf(x, loc=0, scale=10)
M = np.dot(exp_x[:, None], exp_y[None, :])
plt.contour(x, y, M) # 尤度と事前分布の積が事後分布になるとのこと
plt.imshow(M, interpolation='none', origin='lower', cmap=jet, extent=(0, 5, 0, 5))
plt.scatter(lambda_2_true, lambda_1_true, c='k', s=50, edgecolor='none')
plt.xlim(0, 5)
plt.ylim(0, 5)

plt.subplot(224)
plt.contour(x, y, M * L) # 尤度と事前分布の積が事後分布になるとのこと
plt.imshow(M * L, interpolation='none', origin='lower', cmap=jet, extent=(0, 5, 0, 5))
plt.scatter(lambda_2_true, lambda_1_true, c='k', s=50, edgecolor='none')
plt.xlim(0, 5)
plt.ylim(0, 5)

plt.show()

