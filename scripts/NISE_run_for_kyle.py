"""
creates an (d2, w1, w2) "movie" scan

"""


import NISE as n
import os
import numpy as np

def simulate(plot=False):
    if __name__ == '__main__':

        here = os.path.dirname(__file__)
        
        t = n.experiments.trive # module for experiment class k1 - k2 + k2'
        m = n.lib.measure
        H0 = n.hamiltonians.H0
        inhom = n.hamiltonians.params.inhom
        
        Delta_t = 50. # pulse duration [fs]
        slitwidth = 120. # mono resolution [cm-1]
        
        
        # --- set up hamiltonian ----------------------------------------------
        
        H = H0.Omega(wa_central=0, 
                     tau_ag = Delta_t,
                     tau_2aa = Delta_t)
        
        H.TOs = np.array([5])
        
        print([s for s in dir(H) if s[:2] != '__'])
        
        
        # --- set experiment details ------------------------------------------
        
        w1 = t.w1
        w2 = t.w2
        d2 = t.d2 # tau_12
        
        w1.points = np.linspace(-2.5, 2.5, 32) 
        w1.points*= 4 * np.log(2) / Delta_t * 1/(2*np.pi*3e-5)
        w2.points = w1.points.copy()
        d2.points = np.linspace(-2*Delta_t, 4*Delta_t, 16)
        
        t.exp.set_coord(t.d1, 0.) # set tau_22' to zero delay
        t.exp.set_coord(t.ss, Delta_t) # set pulse widths
        t.exp.timestep = 2.0
        
        # time integration starts 
        #   (relative to first pulse arriving)
        t.exp.early_buffer = np.abs(t.d2.points.min()) + Delta_t
        # time to start recording values to array 
        #   (relative to final pulse arriving)
        t.exp.late_buffer = 5 * Delta_t
        
        # dummy object that is neccessary (for no good reason)
        inhom1 = inhom.Inhom() 
        
        m.Mono.slitwidth = slitwidth
        
        
        # --- define scan object and run scan ---------------------------------
        
        def run_if_not_exists(folder, scan, mp=True, plot=False):
            if not os.path.exists(folder): 
                os.makedirs(folder)
                plot=True
            if len(os.listdir(folder)) != 0:
                print('scan has already been run; importing {0}'.format(folder))
                scan = n.lib.scan.Scan._import(folder)
            else: 
                scan.run(autosave=False, mp=mp)
                scan.save(full_name=folder)
        
            measure = m.Measure(scan, m.Mono, m.SLD)
            measure.run(save=False)
        
            if plot:
                measure.plot(1, yaxis=2, zoom=2)
            return scan, measure.pol
        
        scan = t.exp.scan(t.d2, t.w1, t.w2, H=H, inhom_object=inhom1)
        folder = os.path.join(here, 'sim1')
        
        return run_if_not_exists(folder, scan, plot=plot) # to run again delete folder
    else: return None,None
        
scan, pol = simulate(plot=True)



