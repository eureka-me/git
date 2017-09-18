#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/11
# Filename: p13 

__author__ = 'takutohasegawa'
__date__ = "2017/09/11"

from IPython.core.pylabtools import figsize
import scipy.stats as stats
import numpy as np
from matplotlib import pyplot as plt

a = np.linspace(0, 4, 100)
expo = stats.expon
lambda_ = [0.5, 1]
colors = ["#348ABD", "#A60628"]

for l, c in zip(lambda_, colors):
    plt.plot(a, expo.pdf(a, scale=1. / l),  # float型にするために「.」を打っている
             lw=3, color=c, label="$\lambda = %.1f$" % l)
    plt.fill_between(a, expo.pdf(a, scale=1. / l),
                     color=c, alpha=.33)


plt.legend()
plt.ylabel("Probability of density function at $z$")
plt.xlabel("$z$")
plt.ylim(0, 1.2)
plt.title("title")
plt.show()
