#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/17
# Filename: p63 

__author__ = 'takutohasegawa'
__date__ = "2017/09/17"

import pymc as pm
import numpy as np
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize

np.set_printoptions(precision=3, suppress=True)

challenger_data = np.genfromtxt("data/challenger_data.csv",
                                skip_header=1, usecols=[1, 2],
                                missing_values="NA",
                                delimiter=",")

# NaNを削除
challenger_data = challenger_data[~np.isnan(challenger_data[:, 1])]

# 気温データをプロット（1列目）
print(challenger_data)

plt.scatter(challenger_data[:, 0],
            challenger_data[:, 1],
            s=75, color="k", alpha=0.5)

plt.yticks([0, 1])
plt.show()

