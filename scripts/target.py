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


dt = 50.0  # pulse duration (fs)
slitwidth = 120.0  # mono resolution (wn)

nw = 51  # number of frequency points (w1 and w2)
nt = 51  # number of delay points (d2)


# --- workspace -----------------------------------------------------------------------------------

w_central = 1650.0
coupling = 0.0
tau = 50.0


# create experiment
exp = ws.experiment.builtin("trive")
exp.w1.points = (
    np.linspace(-2.5, 2.5, nw) * 4 * np.log(2) / dt * 1 / (2 * np.pi * 3e-5) + w_central
)
exp.w2.points = (
    np.linspace(-2.5, 2.5, nw) * 4 * np.log(2) / dt * 1 / (2 * np.pi * 3e-5) + w_central
)
# exp.w2.points = 0.
exp.d2.points = np.linspace(-2 * dt, 8 * dt, nt)
exp.w1.active = exp.w2.active = exp.d2.active = True
exp.d2.points = 0 * dt
exp.d2.active = False
exp.timestep = 2.0
exp.early_buffer = 2 * dt
exp.late_buffer = 6 * tau

# create hamiltonian
ham = ws.hamiltonian.Hamiltonian(w_central=w_central, coupling=coupling, tau=tau)
ham.recorded_indices = [7, 8]

# do scan
if __name__ == "__main__":
    with wt.kit.Timer():
        scan = exp.run(ham, mp=False)  # 'cpu')

    # use WrightTools api
    data = scan.sig

    data.create_channel(
        name="measured_amplitude",
        values=np.abs(data.channels[0][:] + data.channels[1][:]).sum(axis=-1)[..., None]
    )

    data.transform("w1", "w2")

    out = wt.artists.interact2D(data, channel=-1)
    plt.show()
