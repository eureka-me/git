#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/17
# Filename: p85 

__author__ = 'takutohasegawa'
__date__ = "2017/09/17"

import pymc as pm
import numpy as np
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D


jet = plt.cm.jet
fig = plt.figure()

x = y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(x, y)


plt.subplot(121)
figsize(12.5, 4)
uni_x = stats.uniform.pdf(x, loc=0, scale=5)
uni_y = stats.uniform.pdf(y, loc=0, scale=5)

M = np.dot(uni_x[:, None], uni_y[None, :])
im = plt.imshow(M, interpolation="none", origin="lower",
                cmap=jet, vmax=1, vmin=-.15, extent=(0, 5, 0, 5))
# im = plt.imshow(M, interpolation="none", origin="lower",
#                 cmap=jet, extent=(0, 5, 0, 5))
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title("Overhead view of landscape formed by Uniform priors")

ax = fig.add_subplot(122, projection='3d')
# ax = fig.add_subplot(122)
ax.plot_surface(X, Y, M, cmap=plt.cm.jet, vmax=1, vmin=-.15)
ax.view_init(azim=390)
ax.set_xlabel('value of p_1')
ax.set_ylabel('value of p_2')
ax.set_zlabel('Density')
plt.title('Alternate view of landscape formed by Uniform Priors')
# plt.show()

figsize(12.5, 5)
fig = plt.figure()
plt.subplot(121)

exp_x = stats.expon.pdf(x, scale=3)
exp_y = stats.expon.pdf(x, scale=10)
M = np.dot(exp_x[:, None], exp_y[None, :])
CS = plt.contour(X, Y, M)
im = plt.imshow(M, interpolation='none', origin='lower', cmap=jet, extent=(0, 5, 0, 5))

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X, Y, M, cmap=jet)
ax.view_init(azim=390)
ax.set_xlabel('Value of p_1')
ax.set_ylabel('Value of p_2')
ax.set_zlabel('Density')

plt.show()