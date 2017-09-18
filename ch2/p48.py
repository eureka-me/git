#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/16
# Filename: p48 

__author__ = 'takutohasegawa'
__date__ = "2017/09/16"

import pymc as pm
from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt


# 定数をセット
p_true = 0.05 # 本当のp
N = 5000

occurrences = pm.rbernoulli(p_true, N)

print(occurrences)
print(occurrences.sum())

print("What is ther observed frequency in Group A? %.4f"
      % occurrences.mean())

print("Does the observed frequency equal the true frequency? %s"
      % (occurrences.mean() == p_true))

""""""""""""

# ベルヌーイ分布に従う確率変数を観測済みにする
p = pm.Uniform("p", lower=0, upper=1)
obs = pm.Bernoulli("obs", p, value=occurrences, observed=True)
mcmc = pm.MCMC([p, obs])
mcmc.sample(50000, 2000)

figsize(12.5, 4)
plt.vlines(p_true, 0, 90, linestyle="--",
           label="true p_A (unknown)")
plt.hist(mcmc.trace("p")[:],
         bins=35, histtype="stepfilled", normed=True)

plt.xlabel("value of p_A")
plt.ylabel("density")
plt.legend()
plt.show()
