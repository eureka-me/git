#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/11
# Filename: p12 

__author__ = 'takutohasegawa'
__date__ = "2017/09/11"

from IPython.core.pylabtools import figsize
import scipy.stats as stats
import numpy as np
from matplotlib import pyplot as plt


figsize(12.5, 4)

poi = stats.poisson
lambda_ = [1.5, 4.25]
colors = ["#348ABD", "#A60628"]

a = np.arange(16) #?
plt.bar(a, poi.pmf(a, lambda_[0]), color=colors[0],
        label="$\lambda = %.1f$" % lambda_[0],
        alpha=0.60, edgecolor=colors[0], lw="3")
plt.bar(a, poi.pmf(a, lambda_[1]), color=colors[1],
        label="$\lambda = %.1f$" % lambda_[1],
        alpha=0.60, edgecolor=colors[1], lw="3")

plt.xticks(a + 0.4, a)
plt.legend()
plt.ylabel("Probability of $k$")
plt.xlabel("$k$")
plt.title("title")
plt.show()

