#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/16
# Filename: p41 

__author__ = 'takutohasegawa'
__date__ = "2017/09/16"

import numpy as np
from matplotlib import pyplot as plt
import pymc as pm

data = np.array([10,5])

fixed_variable = pm.Poisson("fxd", 1, value=data, observed=True)
print("value: ", fixed_variable.value)

fixed_variable.random()
print("value: ", fixed_variable.value)

