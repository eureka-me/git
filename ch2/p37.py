#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/16
# Filename: p37 

__author__ = 'takutohasegawa'
__date__ = "2017/09/16"

import pymc as pm

lambda_1 = pm.Exponential("lambda_1", 1)
lambda_2 = pm.Exponential("lambda_2", 1)
tau = pm.DiscreteUniform("tau", lower=0, upper=10)

print("Initialized values...")
print("lambda_2.value = %.3f" % lambda_2.value)
print("lambda_1.value = %.3f" % lambda_1.value)
print("tau.value = %.3f" % tau.value)

lambda_1.random()
lambda_2.random()
tau.random()

print("\n")
print("After calling random() on the variables...")
print("lambda_2.value = %.3f" % lambda_2.value)
print("lambda_1.value = %.3f" % lambda_1.value)
print("tau.value = %.3f" % tau.value)
