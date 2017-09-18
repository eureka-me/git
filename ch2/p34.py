#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/16
# Filename: p34 

__author__ = 'takutohasegawa'
__date__ = "2017/09/16"

import pymc as pm

lambda_ = pm.Exponential("poisson_param", 1)

data_generator = pm.Poisson("data_generator", lambda_)
data_plus_one = data_generator + 1

print("Children of lambda_:")
print(lambda_.children, "\n")

print("Parents of data_generator:")
print(data_generator.parents, "\n")

print("Children of data_generator:")
print(data_generator.children)

print("lambda_.value=", lambda_.value)
print("data_generator.value=", data_generator.value)
print("data_plus_one.value=", data_plus_one.value)

