.. propagation_

Propagation Methods
===================

Propagation methods are the heart of the computation which do the numerical integration of the differential equations.

Python propagators satisfy the signature:

.. code-block:: python

   def propagator_name(t, efields, n_recorded, hamiltonian)
      pass

In CUDA C, the propegators have a different signature, due to limited memory constraints:

.. code-block:: C
   
   __device__
   pycuda::complex<double>* propagator_name(
                                            const double time_start,
                                            const double time_end,
                                            const double dt, 
                                            const int nEFields,
                                            double* efparams,
                                            int* phase_matching,
                                            const int n_recorded,
                                            Hamiltonian ham,
                                            pycuda::complex<double> *out
                                           )


Note that due to the memory constrained nature of the CUDA device code, time and efields are passed as parameters
rather than the arrays themselves.
The Hamiltonian is a struct, defined by each Hamiltonian object that supports CUDA.
It must include fields for ``int nStates``, ``int nRecorded``, and ``int* recorded_indices``, as well as any fields
used in the Hamiltonian-defined ``Hamiltonian_matrix`` method.
Also note that the output is pre-allocated and passed in by reference.

Runge-Kutta
-----------

Currently the only propagation method implemented in ``WrightSim``.
The version implemented here is a second order Runge-Kutta technique.
It is available for both the Python and CUDA implementations.

.. TODO Add citation/mathematical equation
