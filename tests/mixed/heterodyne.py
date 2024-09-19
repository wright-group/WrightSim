"""
look at the phase fringes ddue to interference with each pulse

E1 interference (first column) should modulate along y-axis
E2 interference (second column) should modulate along diagonal
E2p interference (third column) should modulate along x-axis

simulated and driven experiments (top and bottom rows, resp.)
should have the same phase of fringes
"""


import WrightSim as ws
import WrightTools as wt
import numpy as np

dt = 50
nt = 21
wn_to_omega = 2 * np.pi * 3e-5  # cm / s
w_central = 3000
coupling = 0

ham = ws.hamiltonian.Hamiltonian(
    w_central=w_central * wn_to_omega,
    coupling=coupling * wn_to_omega,
)
ham.recorded_elements = [7, 8]

exp = ws.experiment.builtin('trive')
exp.w1.points = w_central  # wn
exp.w2.points = w_central  # wn
exp.d2.points = np.linspace(-10, 10, nt)  # fs
exp.d1.points = np.linspace(-10, 10, nt)
exp.s1.points = exp.s2.points = dt  # fs

exp.d1.active = exp.d2.active = True

exp.timestep = 1
exp.early_buffer = 100.
exp.late_buffer = 300.


def check_phase():
    # TODO: use assertions to check phase
    ...
    

if __name__ == "__main__":
    scan = exp.run(ham, mp=False, windowed=True)

    efields = scan.efields()
    zi0=scan.sig.channels[0][:]
    zi1=scan.sig.channels[1][:]
    sig=zi0+zi1

    driven_sig = 1j * efields.prod(axis=-2)

    # plot amplitude    
    import matplotlib.pyplot as plt
    plt.close('all')
    fig, gs = wt.artists.create_figure(width='single', nrows=2, cols=[1, 1, 1])
    for i in range(scan.npulses):
        lo = efields[..., i, :]
        if scan.pm[i] == 1:
            lo = lo.conjugate()
        diff = (lo * sig).imag.sum(axis=-1)
        driven_diff = (lo * driven_sig).imag.sum(axis=-1)

        axi = plt.subplot(gs[i])
        axi.pcolormesh(exp.d2.points, exp.d1.points, -diff, 
                    vmin=-np.abs(diff).max(),
                    vmax=np.abs(diff).max(),
                    cmap='signed')
        axi2 = plt.subplot(gs[i+3])
        axi2.pcolormesh(exp.d2.points, exp.d1.points, driven_diff,
                        vmin=-np.abs(driven_diff).max(),
                        vmax=np.abs(driven_diff).max(),
                        cmap='signed')
        [ax.grid(True) for ax in [axi, axi2]]

    plt.show()
