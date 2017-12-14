#! /usr/bin/env python3
"""
creates an (d2, w1, w2) "movie" scan
"""


# --- import --------------------------------------------------------------------------------------


import os

import numpy as np

import WrightSim as ws


# --- define --------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))


dt = 50.  # pulse duration (fs)
slitwidth = 120.  # mono resolution (wn)

nw = 32  # number of frequency points (w1 and w2)
nt = 16  # number of delay points (d2)


# --- workspace -----------------------------------------------------------------------------------


# create experiment
exp = ws.experiment.builtin('trive')
exp.w1.points = np.linspace(-2.5, 2.5, nw) * 4 * np.log(2) / dt * 1 / (2 * np.pi * 3e-5)
exp.w2.points = exp.w1.points.copy()
exp.d2.points = np.linspace(-2 * dt, 4 * dt, nt)
exp.w1.active = exp.w2.active = exp.d2.active = True
exp.set_axis('d1', 0)
exp.set_axis('ss', dt)
exp.timestep = 2.

# create hamiltonian
# TODO:

# do scan
# TODO:

# measure and plot
# TODO:
