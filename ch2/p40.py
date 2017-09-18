#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/16
# Filename: p40 

__author__ = 'takutohasegawa'
__date__ = "2017/09/16"

from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
import pymc as pm

figsize(12.5, 4)

lambda_1 = pm.Exponential("lambda_1", 1)
samples = [lambda_1.random() for i in range(20000)]
plt.hist(samples, bins=70, normed=True, histtype="stepfilled")

plt.title("Prior distribution for lambda_1")
plt.xlabel("value")
plt.ylabel("Density")
plt.xlim(0,8)
plt.show()