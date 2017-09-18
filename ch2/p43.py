#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/16
# Filename: p43 

__author__ = 'takutohasegawa'
__date__ = "2017/09/16"

import numpy as np
from matplotlib import pyplot as plt
import pymc as pm

tau = pm.rdiscrete_uniform(0,80)
alpha = 1. / 20.
lambda_1, lambda_2 = pm.rexponential(alpha, 2)

print("tau:",tau)
print("lambda_1:", lambda_1)
print("lambda_2:", lambda_2)

lambda_ = np.r_[lambda_1 * np.ones(tau),
                lambda_2 * np.ones(80 - tau)]
data = pm.rpoisson(lambda_)

plt.bar(np.arange(80), data, color="gray")
plt.bar(tau-1, data[tau-1], color="red", label="user behavior changed")

plt.xlabel("Time(days")
plt.ylabel("Text messages received")
plt.xlim(0, 80)
plt.legend()
plt.show()
