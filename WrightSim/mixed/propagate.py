import numpy as np
from pycuda import driver as cuda

def runge_kutta(t, efields, n_recorded, hamiltonian):
    """
    Evolves the hamiltonian in time, given the three 1d array of E-field points 
    for each E-field and the hamiltonian

    H[pulse_permutations, t, ij_to, ij_from] is a rank 4 array of the hamiltonian 2D 
    matrix for each time point and necessary pulse permutations

    Uses Runge-Kutta method to integrate 
        --> assumes eqaully spaced t-points (t_int is spacing)

    Unlike earlier implementations, delays, phase, etc., must be incorporated
    externally to the function.  Rotating wave may still be specified.
    """
    # can only call on n_recorded and t after efield_object.E is called
    dt = np.abs(t[1]-t[0])
    # extract attributes of the system
    rho_emitted = np.empty((len(hamiltonian.recorded_indices), n_recorded), dtype=np.complex64)

    # H has 3 dimensions: time and the 2 matrix dimensions
    H = hamiltonian.matrix(efields, t)
    # index to keep track of elements of rho_emitted
    emitted_index = 0
    rho_i = hamiltonian.rho.copy()
    for k in range(len(t)-1):
        # now sum over p equivalent pulse permutations (e.g. TRIVE near-
        # diagonal has 2 permutations)
        # calculate delta rho based on previous rho values
        temp_delta_rho = np.dot(H[k], rho_i)
        temp_rho_i = rho_i + temp_delta_rho*dt
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

   # rho_emitted[s,t], s is out_group index, t is time index
    return rho_emitted

muladd_cuda_source = """
__device__ void muladd(pycuda::complex<double>* a, double b, pycuda::complex<double>* c, double d, int len, pycuda::complex<double>* out)
{
    for (int i=0; i<len; i++)
    {
        out[i] = a[i] * b + c[i] * d;
    }
}
"""

dot_cuda_source = """
__device__ void dot(pycuda::complex<double>* mat, pycuda::complex<double>* vec, int len, pycuda::complex<double>* out)
{
    for(int i=0; i<len; i++)
    {
        pycuda::complex<double> sum = pycuda::complex<double>();
        for (int j=0; j<len; j++)
        {
            sum += vec[i] * mat[i + j * len];
        }
        out[i] = sum;
    }
}
"""

pulse_cuda_source = """
#include <math.h>

__device__ void calc_efield_params(double* params, double mu_0, int n)
{
    for(int i=0; i < n; i++)
    {
        //sigma
        params[1 + i*5] /= (2. * sqrt(log(2.)));
        //mu
        params[2 + i*5] -= mu_0;
        //freq
        params[3 + i*5] *= 2 * M_PI * 3e-5;
        //area -> y
        params[0 + i*5] /= params[1 + i*5] * sqrt(2 * M_PI);
    }
}

__device__ void calc_efield(double* params, int* phase_matching,  double t, int n, pycuda::complex<double>* out)
{
    for(int i=0; i < n; i++)
    {
        out[i] = pycuda::exp(-1. * I * (double)(phase_matching[i]) * (params[3 + i*5] * (t - params[2 + i*5]) + params[4 + i*5]));
        out[i] *= params[0 + i*5] * exp(-1 * (t-params[2 + i*5])*(t-params[2 + i*5])/2./params[1 + i*5]/params[1 + i*5]);
    }
}
"""


runge_kutta_cuda_source = """
__device__ pycuda::complex<double>* runge_kutta(const double time_start, const double time_end, const double dt, 
                                               const int nEFields, double* efparams, double mu_0, int* phase_matching,
                                               const int n_recorded, Hamiltonian ham,
                                               pycuda::complex<double> *out)
{
    pycuda::complex<double> *H_cur = (pycuda::complex<double>*)malloc(ham.nStates * ham.nStates * sizeof(pycuda::complex<double>));
    pycuda::complex<double> *H_next = (pycuda::complex<double>*)malloc(ham.nStates * ham.nStates * sizeof(pycuda::complex<double>));

    int out_index = 0;
    int index=0;
    
    int npoints = (int)((time_end-time_start)/dt);

    pycuda::complex<double>* rho_i = (pycuda::complex<double>*)malloc(ham.nStates * sizeof(pycuda::complex<double>));
    pycuda::complex<double>* temp_delta_rho = (pycuda::complex<double>*)malloc(ham.nStates * sizeof(pycuda::complex<double>)); 
    pycuda::complex<double>* temp_rho_i = (pycuda::complex<double>*)malloc(ham.nStates * sizeof(pycuda::complex<double>)); 
    pycuda::complex<double>* delta_rho = (pycuda::complex<double>*)malloc(ham.nStates * sizeof(pycuda::complex<double>)); 
    pycuda::complex<double>* efields = (pycuda::complex<double>*)malloc(nEFields * sizeof(pycuda::complex<double>)); 

    calc_efield_params(efparams, mu_0, nEFields);

    calc_efield(efparams, phase_matching, time_start, nEFields, efields);

    Hamiltonian_matrix(ham, efields, time_start, H_next);
    for(double t = time_start; t < time_end; t += dt)
    {   
        pycuda::complex<double>* temp = H_cur;
        H_cur = H_next;
        H_next = temp;
        calc_efield(efparams, phase_matching, t+dt, nEFields, efields);
        Hamiltonian_matrix(ham, efields, t+dt, H_next);
        dot(H_cur, rho_i, ham.nStates, temp_delta_rho);
        muladd(rho_i, 1., temp_delta_rho, dt, ham.nStates, temp_rho_i);
        dot(H_next, temp_rho_i, ham.nStates, delta_rho);
        muladd(temp_delta_rho, 1., delta_rho, 1., ham.nStates, delta_rho);
        muladd(rho_i, 1., delta_rho, dt/2., ham.nStates, rho_i);

        if(index > npoints - n_recorded)
        {
            for(int i=0; i < ham.nRecorded; i++)
                out[out_index + i * n_recorded] = rho_i[ham.recorded_indices[i]];
            out_index++;
        }
        index++;
    }
    
    dot(H_cur, rho_i, ham.nStates, temp_delta_rho);
    muladd(rho_i, 1., temp_delta_rho, dt, ham.nStates, rho_i);
    for(int i=0; i < ham.nRecorded; i++)
        out[out_index + i * n_recorded] = rho_i[ham.recorded_indices[i]];

    free(H_next);
    free(H_cur);
    free(rho_i);
    free(temp_delta_rho);
    free(temp_rho_i);
    free(delta_rho);
    free(efields);
    return out;
}
"""
