import numpy as np

def runge_kutta(t, efields, n_recorded, hamiltonian):
    """ Evolves the hamiltonian in time using the runge_kutta method.

    Parameters
    ----------
    t : 1-D array of float
        Time points, equally spaced array.
        Shape T, number of timepoints
    efields : ndarray <Complex>
        Time dependant electric fields for all pulses.
        SHape M x T where M is number of electric fields, T is number of time points.
    n_recorded : int
        Number of timesteps to record at the end of the simulation.
    hamiltonian : Hamiltonian
        The hamiltonian object which contains the inital conditions and the 
            function to use to obtain the matrices.

    Returns
    -------
    ndarray : <Complex>
        2-D array of recorded density vector elements for each time step in n_recorded.
    """
    # can only call on n_recorded and t after efield_object.E is called
    dt = np.abs(t[1]-t[0])
    # extract attributes of the system
    rho_emitted = np.empty((len(hamiltonian.recorded_indices), n_recorded), dtype=np.complex128)

    # H has 3 dimensions: time and the 2 matrix dimensions
    H = hamiltonian.matrix(efields, t)
    # index to keep track of elements of rho_emitted
    emitted_index = 0
    rho_i = hamiltonian.rho.copy()
    for k in range(len(t)-1):
        # calculate delta rho based on previous rho values
        temp_delta_rho = np.dot(H[k], rho_i)
        temp_rho_i = rho_i + temp_delta_rho*dt
        # second order term
        delta_rho = np.dot(H[k+1], temp_rho_i)
        rho_i += dt/2 * (temp_delta_rho + delta_rho)
        # if we are close enough to final coherence emission, start 
        # storing these values
        if k >= len(t) - n_recorded:
            for rec,native in enumerate(hamiltonian.recorded_indices):
                rho_emitted[rec, emitted_index] = rho_i[native]
            emitted_index += 1
    # Last timestep
    temp_delta_rho = np.dot(H[-1], rho_i)
    rho_i += temp_delta_rho*dt
    for rec,native in enumerate(hamiltonian.recorded_indices):
        rho_emitted[rec, emitted_index] = rho_i[native]

    return rho_emitted

muladd_cuda_source = """
/*
 * muladd: Linear combination of two vectors with two scalar multiples
 * 
 * computes a*b + c*d where a and c are vectors, b and d are scalars
 * Values are stored in out.
 * If out is the same as a or c, this is an in place operation.
 * len defines the length to perform the computation.
 * Generalizes to n-D array, given it is stored in contiguous memory.
 */
__device__ void muladd(pycuda::complex<double>* a, double b, pycuda::complex<double>* c, double d,
                       int len, pycuda::complex<double>* out)
{
    for (int i=0; i<len; i++)
    {
        out[i] = a[i] * b + c[i] * d;
    }
}
"""

dot_cuda_source = """
/*
 * dot: Matrix multiplied by a vector.
 *
 * Expects a square NxN matrix and an N-length column vector.
 * Values are written to out. DO NOT use in place, invalid results will be returned.
 *
 */
__device__ void dot(pycuda::complex<double>* mat, pycuda::complex<double>* vec, int len,
                    pycuda::complex<double>* out)
{
    for(int i=0; i<len; i++)
    {
        pycuda::complex<double> sum = pycuda::complex<double>();
        for (int j=0; j<len; j++)
        {
            sum += vec[j] * mat[i * len + j];
        }
        out[i] = sum;
    }
}
"""

pulse_cuda_source = """
#include <math.h>

/*
 * calc_efield_params: convert efield params into appropriate values
 * 
 * Converts FWHM to standard deviation
 * Converts frequency into the rotating frame
 * Converts area of peak to height
 *
 * Performs calculaiton on N contiguous sets of paramters, operation in place.
 *
 */
__device__ void calc_efield_params(double* params, int n)
{
    for(int i=0; i < n; i++)
    {
        // FWHM to sigma
        params[1 + i*5] /= (2. * sqrt(log(2.)));
        // Frequency to rotating frame
        params[3 + i*5] *= 2 * M_PI * 3e-5;
        // area -> y
        params[0 + i*5] /= params[1 + i*5] * sqrt(2 * M_PI);
    }
}

/*
 * calc_efield: Convert parameters, phase matching, and time into an electric field
 *
 * Converts n electric fields at a time, places the complex electric
 *      field value into out, in contiguous fashion.
 * The length of the phase_matiching array must be at least n.
 *
 */
__device__ void calc_efield(double* params, int* phase_matching,  double t, int n,
                            pycuda::complex<double>* out)
{
    //TODO: ensure phase matching is done correctly for cases where
    //      it is not equal to +/- 1 (or 0, though why would you have 0)
    //      NISE took the sign, so far I have only taken the value
    for(int i=0; i < n; i++)
    {
        // Complex phase and magnitude
        out[i] = pycuda::exp(-1. * I * ((double)(phase_matching[i]) *
                                        (params[3 + i*5] * (t - params[2 + i*5]) + params[4 + i*5])));
        // Gaussian envelope
        out[i] *= params[0 + i*5] * exp(-1 * (t-params[2 + i*5]) * (t-params[2 + i*5])
                                        / 2. / params[1 + i*5] / params[1 + i*5]);
    }
}
"""


runge_kutta_cuda_source = """
/*
 * runge_kutta: Propagate electric fields over time using Runge-Kutta integration
 * 
 * Parameters
 * ----------
 * time_start: inital simulation time
 * time_end: final simulation time
 * dt: time step
 * nEFields: number of electric fields
 * *efparams: pointer to array of parameters for those electric fields
 * *phase_matiching: array of phase matching conditions
 * n_recorded: number of output values to record
 * ham: Hamiltonian struct containing inital values, passed to matrix generator.
 * 
 * Output:
 * *out: array of recorded values. expects enough memory for n_recorded * ham.nRecorded
 *          complex values
 */
__device__
pycuda::complex<double>* runge_kutta(const double time_start, const double time_end, const double dt, 
                                     const int nEFields, double* efparams, int* phase_matching,
                                     const int n_recorded, Hamiltonian ham,
                                     pycuda::complex<double> *out)
{
    // Allocate arrays and pointers for the Hamiltonians for the current and next step.
    //pycuda::complex<double> *H_cur = (pycuda::complex<double>*)malloc(ham.nStates * ham.nStates * sizeof(pycuda::complex<double>));
    //pycuda::complex<double> *H_next = (pycuda::complex<double>*)malloc(ham.nStates * ham.nStates * sizeof(pycuda::complex<double>));
    //TODO: either figure out why dynamically allocated arrays weren't working, or use a #define to statically allocate
    pycuda::complex<double> buf1[81];
    pycuda::complex<double> buf2[81];

    pycuda::complex<double>* H_cur = buf1;
    pycuda::complex<double>* H_next = buf2;

    // Track indices in arrays.
    int out_index = 0;
    int index=0;
    
    // determine number of points.
    int npoints = (int)((time_end-time_start)/dt);

    // Allocate vectors used in computation.
    pycuda::complex<double>* rho_i = (pycuda::complex<double>*)malloc(ham.nStates * sizeof(pycuda::complex<double>));
    pycuda::complex<double>* temp_delta_rho = (pycuda::complex<double>*)malloc(ham.nStates * sizeof(pycuda::complex<double>)); 
    pycuda::complex<double>* temp_rho_i = (pycuda::complex<double>*)malloc(ham.nStates * sizeof(pycuda::complex<double>)); 
    pycuda::complex<double>* delta_rho = (pycuda::complex<double>*)malloc(ham.nStates * sizeof(pycuda::complex<double>)); 
    pycuda::complex<double>* efields = (pycuda::complex<double>*)malloc(nEFields * sizeof(pycuda::complex<double>)); 

    // Inital rho vector.
    //TODO: Use the inital condition from the hamiltonian
    rho_i[0] = 1.;
    for(int i=1; i<ham.nStates; i++) rho_i[i] = 0.;

    // Convert from given units to simulation units.
    calc_efield_params(efparams, nEFields);

    // Compute the first set of electric fields.
    calc_efield(efparams, phase_matching, time_start, nEFields, efields);

    // Compute the inital matrix, stored in H_next, to be swapped
    Hamiltonian_matrix(ham, efields, time_start, H_next);
    for(double t = time_start; t < time_end; t += dt)
    {   
        // Swap pointers to current and next hamiltonians
        pycuda::complex<double>* temp = H_cur;
        H_cur = H_next;
        H_next = temp;
        
        // First order
        calc_efield(efparams, phase_matching, t+dt, nEFields, efields);
        Hamiltonian_matrix(ham, efields, t+dt, H_next);
        dot(H_cur, rho_i, ham.nStates, temp_delta_rho);
        muladd(rho_i, 1., temp_delta_rho, dt, ham.nStates, temp_rho_i);
        // Second order
        dot(H_next, temp_rho_i, ham.nStates, delta_rho);
        muladd(temp_delta_rho, 1., delta_rho, 1., ham.nStates, delta_rho);
        muladd(rho_i, 1., delta_rho, dt/2., ham.nStates, rho_i);

        // Record results if close enough to the end
        if(index > npoints - n_recorded)
        {
            for(int i=0; i < ham.nRecorded; i++)
            {
                out[out_index + i * n_recorded] = rho_i[ham.recorded_indices[i]];
            }
            out_index++;
        }
        index++;
    }
    
    // Last point, only first order, recorded
    dot(H_cur, rho_i, ham.nStates, temp_delta_rho);
    muladd(rho_i, 1., temp_delta_rho, dt, ham.nStates, rho_i);
    for(int i=0; i < ham.nRecorded; i++)
        out[out_index + i * n_recorded] = rho_i[ham.recorded_indices[i]];

    //free(H_next);
    //free(H_cur);
    free(rho_i);
    free(temp_delta_rho);
    free(temp_rho_i);
    free(delta_rho);
    free(efields);

    return out;
}
"""
