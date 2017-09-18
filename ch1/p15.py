#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/12
# Filename: p15 

__author__ = 'takutohasegawa'
__date__ = "2017/09/12"


from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt


figsize(12.5, 3.5)

count_data = np.loadtxt("./data/txtdata.csv")
n_count_data = len(count_data)
plt.bar(np.arange(n_count_data), count_data)

plt.xlabel("Time(days)")
plt.ylabel("Text messages recieved")
plt.title("title")
plt.xlim(0, n_count_data)
plt.show()
