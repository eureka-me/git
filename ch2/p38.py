#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/16
# Filename: p38 

__author__ = 'takutohasegawa'
__date__ = "2017/09/16"

import pymc as pm

import pymc as pm

lambda_1 = pm.Exponential("lambda_1", 1)
lambda_2 = pm.Exponential("lambda_2", 1)
tau = pm.DiscreteUniform("tau", lower=0, upper=10)

print(type(lambda_1))  # <class 'pymc.distributions.Exponential'>
print(type(lambda_2))  # <class 'pymc.distributions.Exponential'>
print(type(lambda_1+lambda_2))  # <class 'pymc.PyMCObjects.Deterministic'>
