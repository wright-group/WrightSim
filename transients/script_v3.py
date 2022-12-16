import matplotlib.pyplot as plt
from _scan_2dfreq import *
import _transientsv3 as _trans
import numpy as np
import os
from multiprocessing import Process, Queue
import time


omegas = [2253, 3164, 77000] #in wn
gammas = [28, 43, 1000] #in wn
rabis = [2.045e11, 5.817e10, 10e12]
exp = TransientOut(omegas, gammas, rabis)

d1 = 200e-15
d2 = 100e-15
delays = [d1, d2]
exp.set_delays(delays)

pw1 = 200e-15
pw2 = 200e-15
pw3 = 200e-15
pws = [pw1, pw2, pw3]
exp.set_pws(pws)
exp.set_times()
exp.set_pulse_freqs([0, 0, omegas[2]])

w1, w2, width1, width2 = omegas[0], omegas[1], 100, 100
scan_dims = [w1, w2, width1, width2]
shape = (40, 40)

n_threads = 2 #ensure it is multiple of w2 scan width
if n_threads <= os.cpu_count() and __name__ == "__main__":
    spectra = exp.dove_ir_1_freq_scan(scan_dims, shape, n_threads=n_threads)
    print(spectra.scan)

