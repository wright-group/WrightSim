import _transientsv3
from _transientsv3 import *
import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
import timeit
import scipy
from scipy.ndimage import gaussian_filter

# Test script
omega1_trans = wntohz(1600)
omega2_trans = wntohz(3100)
omega3_trans = wntohz(9800)
omega_l1 = wntohz(1600)
omega_l2 = wntohz(3000)
omega_l3 = wntohz(9800)
gamma_gaval = 2e12
gamma_baval = 3e12
gamma_eaval = 10*2e12

Rabi1 = 2.045e12
Rabi2 = 5.817e11
Rabi3 = 9.933e12

d1 = 200e-15
d2 = 200e-15

pw1 = 280e-15
pw2 = 180e-15
pw3 = 600e-15

t0 = 0
t1 = round(t0+pw1, 18)
t2 = t1+d1
t3 = round(t2+pw2, 18)
t4 = t3+d2
t5 = t4+pw3
tmax = 2e-12

t_npts = 1000  # number of points between 1 fs intervals (1000 = time interval per point is is 0.001 fs)

# scan parameters
w1_center = 1600
w2_center = 3100
w1_range = 200
w2_range = 200

scan_npts = 51
w1_scan_range = np.linspace(w1_center-w1_range, w1_center+w1_range, scan_npts)
w2_scan_range = np.linspace(w2_center-w2_range, w2_center+w2_range, scan_npts)

scan = np.zeros((len(w1_scan_range), len(w2_scan_range)))

scan_start = time.time()
remaining = len(w1_scan_range)*len(w2_scan_range)


for ind2, w2 in enumerate(w2_scan_range):  # y axis
    for ind1, w1 in enumerate(w1_scan_range):  # x axis
        time1 = time.time()

        T1 = _transientsv3.bra_abs(Rabi1,
                                   delta_ij(0, 1/1e-12),
                                   delta_ij(omega1_trans, gamma_gaval),
                                   wntohz(w1),
                                   omega1_trans,
                                   gamma_gaval,
                                   1/1e-12,
                                   t1)  # trans1, driven, 0 to t1
        FID1 = _transientsv3.fid(1, delta_ij(omega1_trans, gamma_gaval), t3-t1)
        T2 = _transientsv3.ket_abs(Rabi2,
                                   delta_ij(omega1_trans, gamma_gaval),
                                   delta_ij(omega2_trans, gamma_baval),
                                   wntohz(w2),
                                   omega2_trans,
                                   gamma_baval,
                                   gamma_gaval,
                                   t3-t2)
        FID2 = _transientsv3.fid(1, delta_ij(omega2_trans, gamma_baval),
                                 np.linspace(t4-t3, t5-t3, int((t5-t4)*1e15*t_npts)+1))
        T3 = _transientsv3.ket_abs(Rabi3,
                                   delta_ij(omega2_trans, gamma_baval),
                                   delta_ij(omega3_trans, gamma_eaval),
                                   omega_l3,
                                   omega3_trans,
                                   gamma_eaval,
                                   gamma_baval,
                                   np.linspace(0, t5-t4, int((t5-t4)*1e15*t_npts)+1))

        coeff = T1*FID1*T2
        test = np.array(coeff*FID2*T3)
        scan[ind2][ind1] = np.sum(np.real((test * np.conjugate(test))))

        remaining -= 1

        time2 = time.time()
        print(f'Finished x={w1} and y={w2} | '
              f'Calc. time was {round(time2-time1,2)} s | '
              f'Time remaining is {round((time2-time1)*remaining/60, 2)} min')

scan_end = time.time()
print(f'Total calc. time was {scan_end-scan_start}s')
# print(scan)


#scan = gaussian_filter(scan, sigma=2)
im = plt.imshow(scan, cmap='bwr', origin='lower', extent=(min(w1_scan_range),
                                                     max(w1_scan_range),
                                                     min(w2_scan_range),
                                                     max(w2_scan_range)))

plt.show()
