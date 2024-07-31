"""smokescreen checks of mixed domain default hamiltonian integration"""
import WrightSim as ws
import WrightTools as wt
import numpy as np
import matplotlib.pyplot as plt
import pytest


dt = 20
nt = 21
wn_to_omega = 2 * np.pi * 3e-5  # cm / fs
w_central = 3000  # wn
coupling = 0  # wn

ham = ws.hamiltonian.Hamiltonian(
    w_central=w_central,
    coupling=coupling,
    tau=100,
)
ham.recorded_elements = [7, 8]


@pytest.mark.skip("this test currently fails; bugfix needed")
def test_windowed():
    exp = ws.experiment.builtin('trive')
    exp.w1.points = w_central  # wn
    exp.w2.points = w_central  # wn
    exp.d2.points = 50  # np.zeros((1,))  # fs
    exp.d1.points = 0  # fs
    exp.s1.points = exp.s2.points = dt  # fs

    exp.d1.active = exp.d2.active = False

    # 400 time points
    exp.timestep = 1
    exp.early_buffer = 100.
    exp.late_buffer = 300.

    scan = exp.run(ham, mp=False)
    data = scan.sig

    # shift delay so emission is timed differently
    exp2 = ws.experiment.builtin('trive')
    exp2.w1.points = w_central  # wn
    exp2.w2.points = w_central  # wn
    exp2.d2.points = 50 # np.zeros((1,))  # fs
    exp2.d1.points = 0  # fs
    exp2.s1.points = exp2.s2.points = dt  # fs

    exp2.d1.active = exp2.d2.active = False

    exp2.timestep = 1
    exp2.early_buffer = 100.
    exp2.late_buffer = 300.

    scan2 = exp2.run(ham, mp=False, windowed=True)
    data2 = scan2.sig

    if True:
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.plot(data.time[:], data.channels[0][:].real)

        wn = np.fft.fftfreq(n=data.time.size, d=exp.timestep) / 3e-5
        sig_fft = np.abs(np.fft.fft(data.channels[0][:]))
        ax2.plot(wn, sig_fft)

        ax1.plot(data2.time[:], data2.channels[0][:].real)
        # ax1.plot(data2.time[:], data2.channels[0][:].imag)

        wn2 = np.fft.fftfreq(n=data.time.size, d=exp.timestep) / 3e-5
        sig_fft2 = np.abs(np.fft.fft(data.channels[0][:]))
        ax2.plot(wn2, sig_fft2)

        ax2.set_xlim(-4000, -2000)

        plt.show()

    assert data2.time.size == data.time.size
    assert data2.time.size == data2.channels[0].size
    assert np.all(np.isclose(data2.channels[0][:], data.channels[0][:]))


def test_frequency():

    exp = ws.experiment.builtin('trive')
    exp.w1.points = w_central  # wn
    exp.w2.points = w_central  # wn
    exp.d2.points = 0  # np.zeros((1,))  # fs
    exp.d1.points = 0  # fs
    exp.s1.points = exp.s2.points = dt  # fs

    exp.d1.active = exp.d2.active = False

    # 400 time points
    exp.timestep = 1
    exp.early_buffer = 100.
    exp.late_buffer = 300.

    scan = exp.run(ham, mp=False)
    data = scan.sig
    wn = np.fft.fftfreq(n=data.time.size, d=exp.timestep) / 3e-5
    sig_fft = np.abs(np.fft.fft(data.channels[0][:]))

    if False:
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax2.plot(wn, sig_fft)
        plt.show()

    assert np.abs(wn[np.argmax(sig_fft)] + w_central) < np.abs(wn[1] - wn[0])
   

if __name__ == "__main__":
    test_windowed()  # fails atm
    test_frequency()
