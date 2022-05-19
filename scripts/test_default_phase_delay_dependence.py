import WrightSim as ws
import WrightTools as wt
import numpy as np
import matplotlib.pyplot as plt

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

scan = exp.run(ham, mp=False, windowed=True)

# plot amplitude
if False:
    zi = np.sum(np.abs(np.sum(scan.sig, axis=-2)), axis=-1)
    yi = exp.d1.points
    xi = exp.d2.points
    
    fig, gs = wt.artists.create_figure(cols=[1])
    ax0 = plt.subplot(gs[0])
    ax0.pcolormesh(xi, yi, zi, cmap='default')
    plt.grid(b=True)
    plt.xlabel(exp.d2.name)
    plt.ylabel(exp.d1.name)
    plt.show()

efields = scan.efields()
sig = scan.sig.sum(axis=-2)
driven_sig = 1j * efields.prod(axis=-2)

# look at the phase fringes ddue to interference with each pulse

# E1 interference (first column) should modulate along y-axis
# E2 interference (second column) should modulate along diagonal
# E2p interference (third column) should modulate along x-axis

# simulated and driven experiments (top and bottom rows, resp.) 
# should have the same phase of fringes

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
                   cmap=wt.artists.colormaps['signed'])
    plt.grid(b=True)
    axi2 = plt.subplot(gs[i+3])
    axi2.pcolormesh(exp.d2.points, exp.d1.points, driven_diff,
                    vmin=-np.abs(driven_diff).max(),
                    vmax=np.abs(driven_diff).max(),
                    cmap=wt.artists.colormaps['signed'])
    plt.grid(b=True)

plt.show()
