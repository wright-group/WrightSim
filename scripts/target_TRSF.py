#! /usr/bin/env python3
"""
creates an (wIR, w4) scan
"""


# --- import --------------------------------------------------------------------------------------


import os

import numpy as np

from matplotlib import pyplot as plt

import WrightTools as wt
import WrightSim as ws

# --- define --------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))
wn_to_omega = 2*np.pi*3*10**-5

dt = 1e3  # pulse duration (fs)
slitwidth = 120.  # mono resolution (wn)

nw = 16  # number of frequency points (w1 and w2)

# --- workspace -----------------------------------------------------------------------------------


# create experiment
exp = ws.experiment.builtin('trsf')
exp.w1.points = np.linspace(1400, 1700, nw)
exp.w2.points = exp.w1.points.copy()
exp.w3.points = np.linspace(15500, 17500, nw)

exp.w1.active = exp.w2.active = exp.w3.active = True

exp.timestep = dt/100.
exp.early_buffer = 1.5e3
exp.late_buffer = 1.5e3

# create hamiltonian
ham = ws.hamiltonian.Hamiltonian_TRSF()

# do scan
if __name__ == '__main__':
    scan = exp.run(ham, mp='cpu')

    #"""
    plt.close('all')
    # measure and plot
    fig, gs = wt.artists.create_figure(cols=[1, 'cbar'])
    ax = plt.subplot(gs[0, 0])
    xi = exp.active_axes[0].points
    yi = exp.active_axes[2].points
    
    zi = np.sum(np.abs(np.sum(scan.sig, axis=-2)), axis=-1)
    zi = zi.diagonal(axis1=0, axis2=1).copy()
    zi /= np.nanmax(zi)
    ax.pcolormesh(xi, yi, zi, vmin=0, vmax=1, cmap='default')
    ax.contour(xi, yi, zi, colors='k')
    # decoration
    ax.set_xlabel(exp.active_axes[0].name)
    ax.set_ylabel(exp.active_axes[2].name)
    cax = plt.subplot(gs[0, 1])
    wt.artists.plot_colorbar(label='ampiltude')
    # finish
    plt.show()
    #"""