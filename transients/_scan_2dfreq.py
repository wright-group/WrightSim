__all__ = ['TransientOut']

import numpy as np
import matplotlib.pyplot as plt
import _transientsv3 as _trans
import time
import os
from multiprocessing import Queue, Process


class TransientOut():
    def __init__(self, omegas, gammas, rabis, delays=[], pws=[], times=[], pulse_freqs=[]):
        self.omegas = list(map(_trans.wntohz, omegas))
        self.gammas = list(map(_trans.wntohz, gammas))
        self.rabis = rabis
        self.delays = delays
        self.pws = pws
        self.times = times
        self.pulse_freqs = [0,0,0]######################

    def set_delays(self, delays):
        self.delays = delays
    
    def get_delays(self):
        return self.delays

    def set_pws(self, pws):
        self.pws = pws

    def get_pws(self):
        return self.pws

    def set_times(self):
        if len(self.pws)>len(self.delays)>0:
            t0 = 0
            t1 = round(t0+self.pws[0], 18)
            t2 = t1+self.delays[0]
            t3 = round(t2+self.pws[1], 18)
            t4 = t3+self.delays[1]
            t5 = t4+self.pws[2]
            self.times = [t0,t1,t2,t3,t4,t5]
        else: 
            print('Invalid pulse width and/or delay inputs.')

    def get_times(self):
        return self.times

    def set_pulse_freqs(self, freqs):
        self.pulse_freqs = list(map(_trans.wntohz, freqs))

    def get_pulse_freqs(self):
        return self.pulse_freqs

    def dove_ir_1_freq_scan(self, scan_freqs, npts, time_int=1000, n_threads=1, queue=None):
        # scan parameters
        w1_center = scan_freqs[0]
        w2_center = scan_freqs[1]
        w1_range = scan_freqs[2]
        w2_range = scan_freqs[3]

        w1_scan_range = np.linspace(w1_center-w1_range, w1_center+w1_range, npts[0])
        w2_scan_range = np.linspace(w2_center-w2_range, w2_center+w2_range, npts[1])
        self.w1_scan_range = w1_scan_range
        self.w2_scan_range = w2_scan_range

        self.scan = np.zeros((len(w2_scan_range), len(w1_scan_range)))

        if n_threads==1: 
            scan_start = time.time()
            remaining = len(w1_scan_range)*len(w2_scan_range)
            last_speed = [0]*5
            for ind2, w2 in enumerate(w2_scan_range):  # y axis
                for ind1, w1 in enumerate(w1_scan_range):  # x axis
                    self.set_pulse_freqs([w1, w2, self.omegas[2]])

                    time1 = time.time()

                    ground_gamma = _trans.wntohz(1-18)
                    T1 = _trans.bra_abs(self.rabis[0],
                                            _trans.delta_ij(0, ground_gamma),
                                            _trans.delta_ij(self.omegas[0], self.gammas[0]),
                                            self.pulse_freqs[0],
                                            self.omegas[0],
                                            self.gammas[0],
                                            ground_gamma,
                                            self.times[1])  # trans1, driven, 0 to t1
                    FID1 = _trans.fid(1, _trans.delta_ij(self.omegas[0], self.gammas[0]), self.times[3]-self.times[1])
                    T2 = _trans.ket_abs(self.rabis[1],
                                            _trans.delta_ij(self.omegas[0], self.gammas[0]),
                                            _trans.delta_ij(_trans.wntohz(3164-2253), self.gammas[1]),
                                            _trans.wntohz(w2-w1),
                                            _trans.wntohz(3164-2253),
                                            self.gammas[1],
                                            self.gammas[0],
                                            self.times[3]-self.times[2])
                    FID2 = _trans.fid(1, _trans.delta_ij(_trans.wntohz(3164-2253), self.gammas[1]),
                                            np.linspace(self.times[4]-self.times[3], \
                                            self.times[5]-self.times[3], \
                                            int((self.times[5]-self.times[4])*1e15*time_int)+1))
                    T3 = _trans.ket_abs(self.rabis[2],
                                            _trans.delta_ij(_trans.wntohz(3164-2253), self.gammas[1]),
                                            _trans.delta_ij(_trans.wntohz(3164-2253+9800), self.gammas[2]),
                                            _trans.wntohz(w2-w1+9800),
                                            _trans.wntohz(3164-2253+9800),
                                            self.gammas[2],
                                            self.gammas[1],
                                            np.linspace(0, self.times[5]-self.times[4], int((self.times[5]-self.times[4])*1e15*time_int)+1))

                    coeff = T1*FID1*T2
                    test = np.array(coeff*T3)

                    self.scan[ind2][ind1] = np.sum(np.real((test * np.conjugate(test))))
                    remaining -= 1

                    time2 = time.time()
                    round_time = round(time2-time1, 2)
                    last_speed = last_speed[1:]+[round_time]
                    print(f'Finished w1={w1} and w2={w2} | '
                        f'Calc. time was {round_time} s | '
                        f'Time remaining is {round(np.average(last_speed)*remaining/60, 2)} min')
            scan_end = time.time()
            print(f'Total calc. time was {scan_end-scan_start}s')
            if queue:
                queue.put((w2_center, self.scan))
                return self.scan
            else:
                return self.scan

        if 1<n_threads<=os.cpu_count():
            q = Queue(n_threads)
            w2_range_points = np.linspace(w2_center-w2_range, w2_center+w2_range, n_threads*2+1)
            scan_dims = [(w1_center, point, w2_center, w2_range_points[ind+1]-point)
                        for ind, point in enumerate(w2_range_points) if ind % 2 == 1]
            print(scan_dims)
            shapes = [(npts[0], npts[1]//n_threads)]*n_threads
            subplots = []
            for thread in range(n_threads):
                subexp = TransientOut(self.omegas, self.gammas, self.rabis)
                subexp.set_delays(self.delays)
                subexp.set_pws(self.pws)
                subexp.set_times()
                subexp.set_pulse_freqs([0, 0, self.omegas[2]])
                subplots.append(subexp)

            processes = []
            for ind, sub in enumerate(subplots):
                process = Process(target=sub.dove_ir_1_freq_scan, args=(scan_dims[ind], shapes[ind],), kwargs={'n_threads':1, 'queue':q})
                processes.append(process)
                process.start()
            while not q.full():
                time.sleep(1)

            results = list(q.get() for i in range(q.qsize()))
            sort_results = [i[1] for i in sorted(results, key=lambda x: x[0])]
            self.scan = np.concatenate(sort_results)
            return self


    def plot(self):
        if self.scan.any():
            # scan = gaussian_filter(scan, sigma=2)
            plt.imshow(self.scan, cmap='bwr', origin='lower', extent=(min(self.w1_scan_range),
                                                                      max(self.w1_scan_range),
                                                                      min(self.w2_scan_range),
                                                                      max(self.w2_scan_range)))
            plt.colorbar()
            plt.show()
