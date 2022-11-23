from _scan_2dfreq import *
import _transientsv3 as _trans

omegas = [2254, 3165, 9800]
gammas = list(map(_trans.wntohz, (3.9, 4.7, 0)))
rabis = [2.045e12, 5.817e11, 9.933e12]
exp = TransientOut(omegas, gammas, rabis)

d1 = 200e-15
d2 = 200e-15
delays = [d1, d2]
exp.set_delays(delays)

pw1 = 280e-15
pw2 = 180e-15
pw3 = 600e-15
pws = [pw1, pw2, pw3]
exp.set_pws(pws)
exp.set_times()
exp.set_pulse_freqs([0, 0, omegas[2]])


exp.dove_ir_1_freq_scan([omegas[0], omegas[1], 50, 50], [11, 11])



