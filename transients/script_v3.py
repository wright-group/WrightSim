from _scan_2dfreq import *



omegas = [1600, 3100, 9800]
gammas = [2e12, 3e12, 20e12]
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

exp.dove_ir_1_scan([1600,3100,200,200], 21)



