#! /usr/bin/env python3
"""
creates an (d2, w1, w2) "movie" scan
"""


# --- import --------------------------------------------------------------------------------------


import os
import sys
import time

import numpy as np

from matplotlib import pyplot as plt

import WrightTools as wt
import WrightSim as ws

# --- define --------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))


dt = 50.  # pulse duration (fs)
slitwidth = 120.  # mono resolution (wn)

nw = 51  # number of frequency points (w1 and w2)
nt = 51  # number of delay points (d2)


# --- workspace -----------------------------------------------------------------------------------

w_central = 1650.
coupling = 0.
tau = 50.

# create experiment
exp = ws.experiment.builtin('trive')
exp.w1.points = np.linspace(-2.5, 2.5, nw + 5) * 4 * np.log(2) / dt * 1 / (2 * np.pi * 3e-5) + w_central
exp.w2.points = np.linspace(-2.5, 2.5, nw) * 4 * np.log(2) / dt * 1 / (2 * np.pi * 3e-5) + w_central
#exp.w2.points = 0.
exp.d2.points = np.linspace(-2 * dt, 8 * dt, nt)
exp.w1.active = exp.w2.active = exp.d2.active = True
exp.d2.points = 0 * dt
exp.d2.active = False
exp.timestep = 2.
exp.early_buffer = 2 * dt
exp.late_buffer  = 6 * tau

# create hamiltonian
ham = ws.hamiltonian.Hamiltonian(w_central=w_central, coupling=coupling, tau=tau)
# ham.time_orderings = [6]
ham.recorded_indices = [7,8]

# do scan
begin = time.perf_counter()
scan = exp.run(ham, mp=False)  # 'cpu')
print(time.perf_counter()-begin)
# gpuSig = scan.sig.copy()
 #with wt.kit.Timer():
#    scan2 = exp.run(ham, mp=None)
# cpuSig = scan2.sig.copy()
plt.close('all')
# measure and plot

xi = exp.active_axes[0].points
yi = exp.active_axes[1].points

datao=ws.wtdata.convert(scan)

datao.transform('w1','w2','out')
ch0=datao.channels[0][:]
ch1=datao.channels[1][:]
chsumsq=(ch0+ch1)**2
zi = (chsumsq.sum(axis=-1)).T

zi2 = np.abs(scan.sig.sum(axis=-2)).sum(axis=-1).T


fig, gs = wt.artists.create_figure(cols=[1, 'cbar'])
ax = plt.subplot(gs[0, 0])
ax.pcolor(xi, yi, zi, cmap='default')
ax.contour(xi, yi, zi, colors='k')
# decoration
ax.set_xlabel(exp.active_axes[0].name)
ax.set_ylabel(exp.active_axes[1].name)
cax = plt.subplot(gs[0, 1])
wt.artists.plot_colorbar(label='ampiltude')
# finish
plt.show()


fig2, gs2 = wt.artists.create_figure(cols=[1, 'cbar'])
ax2 = plt.subplot(gs2[0, 0])
ax2.pcolor(xi, yi, zi2, cmap='default')
ax2.contour(xi, yi, zi2, colors='k')
# decoration
ax2.set_xlabel(exp.active_axes[0].name)
ax2.set_ylabel(exp.active_axes[1].name)
cax2 = plt.subplot(gs2[0, 1])
wt.artists.plot_colorbar(label='ampiltude')
# finish
plt.show()
