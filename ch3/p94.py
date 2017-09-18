#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/18
# Filename: p94

import pymc as pm
import numpy as np
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize

wd = 'C:/Users/takut/Documents/study/bayes/'
data = np.loadtxt(wd + 'data/mixture_data.csv', delimiter=",")
p = pm.Uniform("p", 0., 1.)
assignment = pm.Categorical("assignment", [p, 1 -p], size=data.shape[0])

taus = 1. / pm.Uniform("stds", 0, 33, size=2)**2
centers = pm.Normal("centers", [120, 190], [0.01, 0.01], size=2)

@pm.deterministic
def center_i(assignment=assignment, centers=centers):
    return centers[assignment]

@pm.deterministic
def tau_i(assignment=assignment, taus=taus):
    return taus[assignment]

print("p: ", p.value)
print("Random assignments: ", assignment.value[:4], "...")
print("Assigned center: ", center_i.value[:4], "...")
print("Assigned precision: ", tau_i.value[:4], "...")

observations = pm.Normal("obs", center_i, tau_i, value=data, observed=True)
model = pm.Model([p, assignment, taus, centers])
mcmc = pm.MCMC(model)
mcmc.sample(50000, 10000)

"""描画"""
figsize(12.5, 9)
line_width = 1
colors = ["red", "blue"]

"""
p99の描画
center_trace = mcmc.trace("centers")[:]
if center_trace[-1, 0] < center_trace[-1, 1]:
    colors = ["blue", "red"]
    # 大きいほうを赤くしようとしている？

plt.subplot(311)
plt.plot(center_trace[:, 0], label="trace of center 0", c=colors[0], lw=line_width)
#　こんな風にｘ軸の値を指定しないと、ｘ軸はカウントになる！
plt.plot(center_trace[:, 1], label="trace of center 1", c=colors[1], lw=line_width)
leg = plt.legend(loc="upper right")
leg.get_frame().set_alpha(0.7)

plt.subplot(312)
std_trace = mcmc.trace("stds")[:]
plt.plot(std_trace[:, 0], label="trace of standard deviation of cluster 0", c=colors[0], lw=line_width)
plt.plot(std_trace[:, 1], label="trace of standard deviation of cluster 1", c=colors[1], lw=line_width)
plt.legend(loc="upper left")

plt.subplot(313)
p_trace = mcmc.trace("p")[:]
plt.plot(p_trace, label="p: frequency of assingnment to cluster 0", color="green", lw=line_width)
plt.xlabel("Steps")
plt.legend()
plt.show()
"""

"""
# p101の描画
figsize(12.5, 4)

mcmc.sample(100000, progress_bar=True)
center_trace = mcmc.trace("centers", chain=1)[:]
prev_center_trace = mcmc.trace("centers", chain=0)[:]

x = np.arange(50000)
plt.plot(x, prev_center_trace[:, 0], label="previous trace of center 0", lw=line_width, alpha=0.4, c=colors[0])
plt.plot(x, prev_center_trace[:, 1], label="previous trace of center 1", lw=line_width, alpha=0.4, c=colors[1])

x = np.arange(50000, 150000)
plt.plot(x, center_trace[:, 0], label="new trace
of center 0", lw=line_width, c=colors[0])
plt.plot(x, center_trace[:, 1], label="new trace of center 1", lw=line_width, c=colors[1])

leg = plt.legend(loc="upper right")
leg.get_frame().set_alpha(0.8)
plt.show()

"""

"""
# p102の描画
figsize(11.0, 4)
center_trace = mcmc.trace("centers")[:]
std_trace = mcmc.trace("stds")[:]

_i = [1, 2, 3, 4]
for i in range(2):
    plt.subplot(2, 2, _i[2 * i])
    plt.hist(center_trace[:, i], color=colors[i], bins=30)

    plt.subplot(2, 2, _i[2 * i + 1])
    plt.hist(std_trace[:, i], color=colors[i], bins=30)

plt.tight_layout()
plt.show()

"""

"""
# p103の描画
import matplotlib as mpl
figsize(12.5, 4.5)

plt.cmap = mpl.colors.ListedColormap(colors)
plt.imshow(mcmc.trace("assignment")[::400, np.argsort(data)],
           cmap=plt.cmap, aspect=.4, alpha=.9)

plt.xticks(np.arange(0, data.shape[0], 40),
           ["%.2f" % s for s in np.sort(data)[::40]])
plt.show()

"""


"""
# p103_2の描画
import matplotlib as mpl
cmap = mpl.colors.LinearSegmentedColormap.from_list("BMH", colors)

assign_trace = mcmc.trace("assignment")[:]
plt.scatter(data, 1 - assign_trace.mean(axis=0), cmap=cmap, c=assign_trace.mean(axis=0), s=50)

plt.ylim(-.05, 1.05)
plt.xlim(35, 300)
plt.show()
"""

# p105の描画
from scipy import stats
norm = stats.norm

x = np.linspace(20, 300, 500)
center_trace = mcmc.trace("centers")[:]
std_trace = mcmc.trace("stds")[:]
posterior_center_means = center_trace.mean(axis=0)
posterior_std_means = std_trace.mean(axis=0)
posterior_p_mean = mcmc.trace("p")[:].mean()

plt.hist(data, bins=20, histtype="step", normed=True, color="k", lw=2, label="histgram of data")

y = posterior_p_mean * norm.pdf(x, loc = posterior_center_means[0], scale=posterior_std_means[0])
plt.plot(x, y, lw=3, color=colors[0], label="cluster 0")
plt.fill_between(x, y, color=colors[0], alpha = 0.3)

y = (1 - posterior_p_mean) * norm.pdf(x, loc = posterior_center_means[1], scale=posterior_std_means[1])
plt.plot(x, y, lw=3, color=colors[1], label="cluster 1")
plt.fill_between(x, y, color=colors[1], alpha = 0.3)

plt.legend(loc="upper left")
plt.show()




