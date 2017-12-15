
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
        # test if index is out of range
        delta_rho = np.dot(H[k+1], temp_rho_i)
        # iterative method fails on the last timestep, so handle it
        rho_i += dt/2 * (temp_delta_rho + delta_rho)
        # if we are close enough to final coherence emission, start 
        # storing these values
        if k >= len(t) - n_recorded:
            for i in hamiltonian.recorded_indices:
                rho_emitted[i, emitted_index] = rho_i[i]
            emitted_index += 1
    # Last timestep
    temp_delta_rho = np.dot(H[-1], rho_i)
    rho_i += temp_delta_rho*dt
    for i in hamiltonian.recorded_indices:
        rho_emitted[i, emitted_index] = rho_i[i]

   # rho_emitted[s,t], s is out_group index, t is time index
    return rho_emitted
