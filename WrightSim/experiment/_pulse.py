# --- import --------------------------------------------------------------------------------------


import numpy as np


# --- define --------------------------------------------------------------------------------------


wn_to_omega = 2 * np.pi * 3e-5  # omega is radians / fs


# factor used to convert FWHM to stdev for function definition
# gaussian defined such that FWHM,intensity scale = sigma * 2 * sqrt(ln(2))
# (i.e. FWHM, intensity scale ~ 1.67 * sigma, amplitude scale)
FWHM_to_sigma = 1. / (2 * np.sqrt(np.log(2)))


# TODO: these should probably not be hard-coded
timestep = 4.0
early_buffer = 100.0
late_buffer  = 400.0

# --- functions -----------------------------------------------------------------------------------




# --- class ---------------------------------------------------------------------------------------


class GaussRWA:
    """
    returns an array of gaussian values with arguments of
        input array vec,
        mean mu,
        standard deviation sigma
        amplitude amp
        frequency freq
        phase offset p

    we are in rw approx, so if interaction is supposed to be negative, give
    frequency a negative sign
    """
    # dictionary to associate array position to pulse attribute
    cols = ['A',  # amplitude, a.u.
            's',  # pulse FWHM (fs),
            'd',  # pulse center delay (fs),
            'w',  # frequency (wn),
            'p']  # phase shift (radians)

    @classmethod
    def pulse(cls, eparams, t_args, pm=None):
        """Get efields for each pulse.

        Parameters
        ----------
        eparams : 2D numpy ndarray
            Array in (pulse, parameter). Refer to cols attribute for list
            of parameters.
        pm : iterable of integers
            Phase matching. If None, fully positive phase matching is
            assumed.

        Returns
        -------
        (1D array, 1D array)
            Tuple of 1D numpy ndarray time (fs) and complex efield (a.u.).
        """
        # import if the size is right
        area  = eparams[:, 0].copy().astype(float)
        sigma = eparams[:, 1].copy().astype(float)
        mu    = eparams[:, 2].copy().astype(float)
        freq  = eparams[:, 3].copy().astype(float)
        p     = eparams[:, 4].copy().astype(float)
        # proper unit conversions
        sigma *= FWHM_to_sigma
        freq *= wn_to_omega
        # redefine delays s.t. always relative to the first pulse listed
        offset = mu[0]
        # subtract off the value
        mu -= offset
        # normalize amplitdue to stdev
        y = area / (sigma * np.sqrt(2 * np.pi))
        #print y, sigma, mu, freq, p
        # calculate t
        # t = cls.get_t(mu)
        t = np.arange(*t_args)
        # incorporate complex conjugates if necessary
        if pm is None:
            cc = np.ones((eparams.shape[-1]))
        else:
            cc = np.sign(pm)
        x = np.exp(-1j * cc[:, None] * (freq[:, None] * (t[None, :] - mu[:, None]) + p[:, None]))
        x*= y[:,None] * np.exp(-(t[None,:] - mu[:,None])**2 / (2*sigma[:,None]**2) )
        return x
