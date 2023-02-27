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
wn_to_omega = 2 * np.pi * 3 * 10**-5

dt = 1e3  # pulse duration (fs)
slitwidth = 120.0  # mono resolution (wn)

nw = 16  # number of frequency points (w1 and w2)

# --- workspace -----------------------------------------------------------------------------------


# create experiment
exp = ws.experiment.builtin("trsf")
exp.w1.points = np.linspace(1400, 1700, nw)
exp.w2.points = exp.w1.points.copy()
exp.w3.points = np.linspace(15500, 17500, nw)

exp.w1.active = exp.w2.active = exp.w3.active = True

exp.timestep = dt / 100.0
exp.early_buffer = 1.5e3
exp.late_buffer = 1.5e3

# create hamiltonian
ham = ws.hamiltonian.Hamiltonian_TRSF()

# do scan
if __name__ == "__main__":
    with wt.kit.Timer():
        scan = exp.run(ham, mp=None)  # "cpu")

    # measure and plot
    data = scan.sig

    data.create_channel(
        name="measured_amplitude",
        values=np.abs(data.channels[0][:] + data.channels[1][:]).sum(axis=-1)[..., None]
    )

    data.transform("w1", "w2", "w3")

    out = wt.artists.interact2D(data, channel=-1)
    plt.show()
