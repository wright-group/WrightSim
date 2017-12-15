# --- import --------------------------------------------------------------------------------------


import numpy as np

from scipy.signal import convolve2d

import WrightTools as wt

from . import pulse as pulse


# --- functions -----------------------------------------------------------------------------------


def do_work(arglist):
    indices, iprime, inhom_object, H, pulse_class = arglist[:5]
    efpi, pm, timestep, eb, lb, evolve_func = arglist[5:]
    # need to declare these for each function
    pulse_class.early_buffer = eb
    pulse_class.late_buffer = lb
    pulse_class.timestep = timestep
    t, efields = pulse_class.pulse(efpi, pm=pm)
    out = evolve_func(t, efields, iprime, inhom_object, H)
    if indices[-1] == 0:
        print(indices, pulse_class.timestep, iprime + '              \r',)
    return indices, out


# --- class ---------------------------------------------------------------------------------------


class Scan:

    def __init__(self, experiment, hamiltonian):
        self.exp = experiment
        self.ham = hamiltonian
        # unpack experiment
        self.axis_objs = self.exp.active_axes
        pulse_class = self.exp.pulse_class
        self.cols = pulse_class.cols
        self.inv_cols = {v: k for k, v in pulse_class.cols.items()}
        self.npulses = len(self.exp.pulses)
        self.pm = self.exp.pm
        self.early_buffer = self.exp.early_buffer
        self.late_buffer = self.exp.late_buffer
        self.timestep = self.exp.timestep
        # initialize
        self.coords_set = []
        self.iprime = np.arange(-self.early_buffer, self.late_buffer, self.timestep).size
        self.shape = tuple(a.points.size for a in self.exp.active_axes)
        self.array = np.zeros(self.shape)
        self.efp = self._gen_efp()

    def _gen_efp(self, indices=None):
        """Get an array containing all parameters of efields.

        Parameters
        ----------
        indicies : array of integers (optional)
            Specific indicies to look up parameters for. If None, all
            indicies are looked up. Default is None.

        Returns
        -------
        numpy ndarray
            Array in (axes..., pulse, parameter).
        """
        if indices is None:
            indices = np.ndindex(self.array.shape)
        shape = list(self.array.shape)
        # add to each dims element a nxm array to store m params for all n pulses
        shape.append(self.npulses)
        num_args = len(self.cols)
        shape.append(num_args)
        efp = np.zeros(shape)
        for index in indices:
            this_point = np.zeros(shape[-2:])
            # loop through each axis to set points
            for i in range(len(index)):
                # for each axis, loop through all vars changed by it
                for j in range(len(self.axis_objs[i].coords)):
                    coords = self.axis_objs[i].coords[j]
                    this_point[coords[0], coords[1]] = self.axis_objs[i].points[index[i]]
            efp[index] = this_point
        # now, if we didn't fill in a value by going through the scan ranges,
        # we now have to fill in with specified constants
        for pi in range(self.npulses):
            for arg in range(num_args):
                indices = [pi, arg]
                if indices in self.coords_set:
                    pass
                else:
                    # fill with default values, but where do i get the default
                    # values from?
                    default_val = self.positions[pi, arg]
                    efp[..., pi, arg] = default_val
        return efp

    def run(self, mp=True, chunk=False):
        """Run the scan.

        Parameters
        ----------
        mp : boolean (optional)
            Toggle CPU multiprocessing. Default is True.

        Returns
        numpy ndarray
            Array in (axes..., outgroups, time)
        """
        shape = list(self.array.shape)
        shape.append(len(self.H.out_group))
        shape.append(self.iprime)
        self.pulse_class.timestep = self.timestep
        self.pulse_class.early_buffer = self.early_buffer
        self.pulse_class.late_buffer = self.late_buffer
        self.pulse_class.pm = self.pm
        self.sig = np.empty(shape, dtype=np.complex64)
        if mp:
            from multiprocessing import Pool, cpu_count
            arglist = [[ind, self.iprime, self.inhom_object, self.H,
                        pulse_class, self.efp[ind], self.pm, self.timestep,
                        self.early_buffer, self.late_buffer, self.ham.propagator]
                        for ind in np.ndindex(self.array.shape)]
            pool = Pool(processes=cpu_count())
            chunksize = int(self.array.size / cpu_count())
            print('chunksize:', chunksize)
            with wt.kit.Timer():
                results = pool.map(do_work, arglist, chunksize=chunksize)
                pool.close()
                pool.join()
            # now write to the np array
            for i in range(len(results)):
                self.sig[results[i][0]] = results[i][1]
            del results
        else:
            with wt.kit.Timer():
                for indices in np.ndindex(self.array.shape):
                    t, efields = pulse_class.pulse(self.efp[indices], pm=self.pm)
                    self.sig[indices] = self.ham.propagator(t, efields, self.iprime, self.ham)
        return self.sig

    def get_color(self):
        """Get an array of driven signal frequency for each array point."""
        # in wavenumbers
        w_axis = self.cols['w']
        wtemp = self.efp[..., w_axis].copy()
        wtemp *= self.pm
        wm = wtemp.sum(axis=-1)
        return wm

    def efields(self, windowed=True):
        """Return the e-fields used in the simulation.

        Parameters
        ----------
           windowed : boolean (optional)
               If True, only returns values that are within the early and
               late buffer. Default is True.

        Returns
        -------
        numpy ndarray
            Array in (axes..., pulse, time).
        """
        # [axes..., numpulses, nparams]
        efp = self.get_efield_params()
        # [axes..., numpulses, pulse field values]
        efields_shape = list(efp.shape)
        if windowed:
            efields_shape[-1] = self.iprime
            efields = np.zeros((efields_shape), dtype=np.complex)
            with wt.kit.Timer():
                for ind in np.ndindex(tuple(efields_shape[:-2])):
                    ti, efi = self.pulse_class.pulse(efp[ind], pm=self.pm)
                    efields[ind] = efi[:, -self.iprime:]
        else:
            # figure out the biggest array size we will get
            d_ind = self.pulse_class.cols['d']
            t = self.pulse_class.get_t(efp[..., d_ind])
            # now that we know t vals, we can set fixed bounds
            self.pulse_class.fixed_bounds_min = t.min()
            self.pulse_class.fixed_bounds_max = t.max()
            self.pulse_class.fixed_bounds = True
            efields_shape[-1] = t.size
            efields = np.zeros((efields_shape), dtype=np.complex)
            try:
                with wt.kit.Timer():
                    for ind in np.ndindex(tuple(efields_shape[:-2])):
                        ti, efi = self.pulse_class.pulse(efp[ind], pm=self.pm)
                        efields[ind] = efi
            finally:
                # set the class back to what it was before exiting
                self.pulse_class.fixed_bounds = False
        return efields
