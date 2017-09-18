#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/11
# Filename: p9 

__author__ = 'takutohasegawa'
__date__ = "2017/09/11"

from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline
figsize(12.5, 4)

colors = ["#348ABD", "#A60628"]
prior = [1/21., 20/21.]
posterior = [0.087,  1 - 0.087]
plt.bar([0, .7], prior, alpha = 0.70, width=0.25,
        color=colors[0], label="prior distribution",  # 事前確率
        lw="3", edgecolor="#348ABD")

plt.bar([0 + 0.25, .7 + 0.25], posterior, alpha=0.7,
        width=0.25, color=colors[1],
        label="posterior distribution",  # 事後確率
        lw="3", edgecolor="#A60628")

plt.xticks([0.20, 0.95], ["Librarian", "Farmer"])  # 司書、農家
plt.ylabel("Probability")
plt.legend(loc="upper left")
plt.title("Prior and posterior probabilities of Steve's occupation")
plt.show()


