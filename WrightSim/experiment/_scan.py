# --- import --------------------------------------------------------------------------------------


import collections

import numpy as np

import WrightTools as wt
from ..mixed import propagate


# --- functions -----------------------------------------------------------------------------------


def do_work(arglist):
    indices, iprime, H, pulse_class = arglist[:4]
    efpi, pm, timestep, eb, lb, evolve_func = arglist[4:]
    # need to declare these for each function
    pulse_class.early_buffer = eb
    pulse_class.late_buffer = lb
    pulse_class.timestep = timestep
    t, efields = pulse_class.pulse(efpi, pm=pm)
    out = evolve_func(t, efields, iprime, H)
    #if indices[-1] == 0:
    #   print(indices, pulse_class.timestep, str(iprime) + '              \r',)
    return indices, out


# --- class ---------------------------------------------------------------------------------------


class Scan:

    def __init__(self, experiment, hamiltonian):
        self.exp = experiment
        self.ham = hamiltonian
        # unpack experiment
        self.axis_objs = self.exp.active_axes
        self.pulse_class = self.exp.pulse_class
        self.cols = self.pulse_class.cols
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
        efp = np.zeros(self.shape + (self.npulses, len(self.cols)))
        for pulse_index in range(self.npulses):
            axes = [a for a in self.exp.axes if pulse_index in a.pulses]
            for axis in axes:
                parameter_index = self.cols.index(axis.parameter)
                if axis.active:
                    axis_index = self.exp.active_axes.index(axis)
                    points = axis.points.copy()
                    for _ in range(axis_index, len(self.exp.active_axes) - 1):
                        points.shape += (1,)
                    efp[..., pulse_index, parameter_index] = points
                else:
                    efp[..., pulse_index, parameter_index] = axis.points
        return efp

    kernel_cuda_source = """
    __global__ void kernel(double time_start, double time_end, double dt, int nEFields, double* efparams, int* phase_matching, int n_recorded, Hamiltonian* ham, pycuda::complex<double>* out)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        runge_kutta(time_start, time_end, dt, nEFields, efparams + (idx*5*nEFields), *(efparams + 2), phase_matching, n_recorded, *ham, out + (idx*ham->nRecorded*n_recorded));
    }
    """

    def run(self, mp='cpu', chunk=False):
        """Run the scan.

        Parameters
        ----------
        mp : {False, 'cpu', 'gpu'} (optional)
            Select multiprocessing: False (or '' or None) means single-threaded.
                                    'gpu' indicates to use the CUDA implementation
                                    Any other value which evaluates to `True` indicates cpu multiprocessed.
                                    Default is 'cpu'.

        Returns
        numpy ndarray
            Array in (axes..., outgroups, time)
        """
        shape = list(self.array.shape)
        shape.append(len(self.ham.recorded_indices))
        shape.append(self.iprime)
        self.pulse_class.timestep = self.timestep
        self.pulse_class.early_buffer = self.early_buffer
        self.pulse_class.late_buffer = self.late_buffer
        self.pulse_class.pm = self.pm
        self.sig = np.empty(shape, dtype=np.complex128)
        if mp == 'gpu':
            from pycuda import driver as cuda
            from pycuda.compiler import SourceModule
            from pycuda import autoinit
            hamPtr = cuda.mem_alloc(self.ham.cuda_mem_size)
            self.ham.to_device(hamPtr)
            efpPtr = cuda.to_device(self.efp)
            pmPtr = cuda.to_device(np.array(self.pm, dtype=np.int32))
            sigPtr = cuda.mem_alloc(self.sig.nbytes)

            d_ind = self.pulse_class.cols.index('d')
            start = np.min(self.efp[..., d_ind]) - self.early_buffer
            stop = np.max(self.efp[..., d_ind]) + self.late_buffer

            mod = SourceModule(self.ham.cuda_struct + self.ham.cuda_matrix_source + propagate.muladd_cuda_source + propagate.dot_cuda_source + propagate.pulse_cuda_source + propagate.runge_kutta_cuda_source + Scan.kernel_cuda_source)

            kernel = mod.get_function('kernel')
            kernel(start, stop, np.float64(self.timestep), np.intp(3), efpPtr, pmPtr, np.intp(self.iprime), hamPtr, sigPtr, grid=(self.array.size//256,1), block=(256,1,1))

            cuda.memcpy_dtoh(self.sig, sigPtr)
        elif mp:
            from multiprocessing import Pool, cpu_count
            arglist = [[ind, self.iprime, self.ham,
                        self.pulse_class, self.efp[ind], self.pm, self.timestep,
                        self.early_buffer, self.late_buffer, self.ham.propagator]
                        for ind in np.ndindex(self.array.shape)]
            pool = Pool(processes=cpu_count())
            chunksize = int(self.array.size / cpu_count())
            #print('chunksize:', chunksize)
            #with wt.kit.Timer():
            results = pool.map(do_work, arglist, chunksize=chunksize)
            pool.close()
            pool.join()
            # now write to the np array
            for i in range(len(results)):
                self.sig[results[i][0]] = results[i][1]
            del results
        else:
            #with wt.kit.Timer():
                for idx in np.ndindex(self.shape):
                    t, efields = self.pulse_class.pulse(self.efp[idx], pm=self.pm)
                    self.sig[idx] = self.ham.propagator(t, efields, self.iprime, self.ham)
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
        efp = self.efp
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
