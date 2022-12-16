import matplotlib.pyplot as plt

from _scan_2dfreq import *
import _transientsv3 as _trans
import numpy as np
import os
from multiprocessing import Process, Queue
import time


omegas = [2253, 3164, 77000] #in wn
gammas = [2.8, 4.3, 1000] #in wn
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

n_threads = 20 #ensure it is multiple of w2 scan width

mp = True

if mp and n_threads <= os.cpu_count() and __name__ == "__main__":
    q = Queue(n_threads)
    w2_range_points = np.linspace(w2-width2, w2+width2, n_threads*2+1)
    scan_dims = [(w1, point, width1, w2_range_points[ind+1]-point)
                 for ind, point in enumerate(w2_range_points) if ind % 2 == 1]
    print(scan_dims)
    shapes = [(shape[0], shape[1]//n_threads)]*n_threads
    subplots = []
    for thread in range(n_threads):
        subexp = TransientOut(omegas, gammas, rabis)
        subexp.set_delays(delays)
        subexp.set_pws(pws)
        subexp.set_times()
        subexp.set_pulse_freqs([0, 0, omegas[2]])
        subplots.append(subexp)

    t1 = time.time()
    processes = []
    for ind, sub in enumerate(subplots):
        process = Process(target=sub.dove_ir_1_freq_scan, args=(scan_dims[ind], shapes[ind],), kwargs={'queue':q})
        processes.append(process)
        process.start()
    while not q.full():
        time.sleep(1)
    t2 = time.time()
    print(f'took {t2-t1} seconds.')

    results = list(q.get() for i in range(q.qsize()))
    sort_res = [i[1] for i in sorted(results, key=lambda x: x[0])]
    spectra = np.concatenate(sort_res)

    plt.imshow(spectra, cmap='bwr', origin='lower', extent=(w1-width1,
                                                            w1+width1,
                                                            w2-width2,
                                                            w2+width2))
    plt.colorbar()
    plt.show()


elif not mp:
    exp.dove_ir_1_freq_scan(scan_dims, shape)
    exp.plot()
