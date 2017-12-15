"""
Pulse module:  for defining pulse classes to be used in simulations
Rules for creating a pulse class method:

1.  "pulse" method:  
    accepts:
        eparams
        pm
    returns:    
        t
        x                
2.  "get_t" method:  should be inherited
    accepts:
        d
    returns:
        t
"""


### import ####################################################################


import numpy as np


### define ####################################################################


wn_to_omega = 2*np.pi*3*10**-5  # omega is radians / fs


# factor used to convert FWHM to stdev for function definition
# gaussian defined such that FWHM,intensity scale = sigma * 2 * sqrt(ln(2))
# (i.e. FWHM, intensity scale ~ 1.67 * sigma, amplitude scale)
FWHM_to_sigma = 1./(2*np.sqrt(np.log(2)))


# TODO: these should probably not be hard-coded
timestep = 4.0
early_buffer = 100.0
late_buffer  = 400.0


### pulse #####################################################################


def _get_t(obj, d):
    """
        returns the t array that is appropriate for the given pulse 
        parameters; needs the appropriate buffers/timestep as well 
    """
    if obj.fixed_bounds:
        t_min = obj.fixed_bounds_min
        t_max = obj.fixed_bounds_max
    else:
        t_min = d.min() - obj.early_buffer
        t_max = d.max() + obj.late_buffer
    # span up to and including t_max now
    t = np.arange(t_min, t_max+obj.timestep, obj.timestep)
    # alternate form:
    return t


class GaussChirpRWA:
    """
    chirped gaussian pulse in rwa
    pulse center is a and chirp is centered around gaussian peak
    returns an array of gaussian values with arguments of 
        mu: mean 
        s:  standard deviation sigma
        A:  amplitude amp
        w:  frequency freq
        p:  (constant) phase offset
        dz: (linear) phase evolution of spectral phase (centered about w) 
            for substantial chirp (~100fs FWHM), try p1 ~ 1e-5
    
    we are in rw approx, so if interaction is supposed to be negative, give
    frequency a negative sign
    """

    cols = {
        'A' : 0, # amplitude, a.u.
        's' : 1, # pulse FWHM (in fs)
        'd' : 2, # pulse center delay fs
        'w' : 3, # frequency in wavenumbers
        'p' : 4, # phase shift, in radians
        'dz': 5  # coefficient for extent of chirp inducement
    }
    # chirp parameter--currently scaled such that chirp extent is considerable
    # (something like doubled peak width) when dz ~ 1e-5
    # positive dz means blue part of the pulse arrives first
    defaults = [1.0, 55., 0., 7000., 0.,0.]
    # initial vars come from misc module, just as with scan module
    timestep = timestep
    early_buffer = early_buffer
    late_buffer = late_buffer
    # fixed bounds set to true if you want fixed time indices for the pulses
    fixed_bounds = False
    fixed_bounds_min = None
    fixed_bounds_max = None

    @classmethod
    def pulse(cls, eparams, pm=None):
        # import if the size is right
        area  = eparams[:,0].copy().astype(float)
        sigma = eparams[:,1].copy().astype(float)
        mu    = eparams[:,2].copy().astype(float)
        freq  = eparams[:,3].copy().astype(float)
        p     = eparams[:,4].copy().astype(float)
        dz    = eparams[:,5].copy().astype(float)
        # proper unit conversions
        sigma *= FWHM_to_sigma
        freq *= wn_to_omega
        # redefine delays s.t. always relative to the first pulse listed
        offset = mu[0]
        # subtract off the value
        mu -= offset
        # normalize amplitdue to stdev
        y = area / (sigma*np.sqrt(2*np.pi))
        #print y, sigma, mu, freq, p
        # calculate t
        t = cls.get_t(mu)
        # incorporate complex conjugates if necessary
        if pm is None:
            cc = np.ones((eparams.shape[-1]))
        else:
            cc = np.sign(pm)
        env = y[:,None] * np.exp(-(t[None,:] - mu[:,None])**2 
                                 / (2*sigma[:,None]**2))
        #print np.abs(env).max()
        env = cls.prop(env, dz, mu)
        #print np.abs(env).max()
        phase = cc[:,None]*(freq[:,None]*(t[None,:] - mu[:,None]) + p[:,None])
        x = env * np.exp(-1j*phase)
        #print np.abs(x).max()
        return t, x

    @classmethod
    def prop(cls, env, dz, mu):
        """
            takes the E-field envelope in time domain and modify it in the 
            frequency domain 
            E:  E(t) as 1D numpy array.  time is in units of fs
            dz:  increment in z(mm) to allow this envelope to propagate
        """
        env_fft = np.fft.ifft(env)
        w = np.fft.fftfreq(env_fft.shape[-1], cls.timestep) 
        wn = w * 2*np.pi / wn_to_omega
        # positive slope so positive dz parameter means normal chirp
        n = wn
        n -= n[0]
        env_fft *= np.exp(1j*n*dz[:,None]*wn[None,:])
        env_new = np.fft.fft(env_fft)
        return env_new

    @classmethod
    def get_t(cls, d):
        return _get_t(cls,d)


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
    # go to values for arguments
    defaults = [1.0, 55., 0., 7000., 0.]
    # dictionary to associate array position to pulse attribute
    cols = {
        'A' : 0, # amplitude, a.u.
        's' : 1, # pulse FWHM (in fs)
        'd' : 2, # pulse center delay fs
        'w' : 3, # frequency in wavenumbers
        'p' : 4  # phase shift, in radians
    }
    # initial vars come from misc module, just as with scan module
    timestep = timestep
    early_buffer = early_buffer
    late_buffer = late_buffer
    # fixed bounds set to true if you want fixed time indices for the pulses
    fixed_bounds = False
    fixed_bounds_min = None
    fixed_bounds_max = None
        
    @classmethod
    def pulse(cls, eparams, pm=None):
        """
            accepts a 2d array where the final index is the params for the 
            fields to be generated
        """
        # import if the size is right
        area  = eparams[:,0].copy().astype(float)
        sigma = eparams[:,1].copy().astype(float)
        mu    = eparams[:,2].copy().astype(float)
        freq  = eparams[:,3].copy().astype(float)
        p     = eparams[:,4].copy().astype(float)
        
        # proper unit conversions
        sigma *= FWHM_to_sigma
        freq *= wn_to_omega
        # redefine delays s.t. always relative to the first pulse listed
        offset = mu[0]
        # subtract off the value
        mu -= offset
        # normalize amplitdue to stdev
        y = area / (sigma*np.sqrt(2*np.pi))
        #print y, sigma, mu, freq, p
        # calculate t
        t = cls.get_t(mu)
        # incorporate complex conjugates if necessary
        if pm is None:
            cc = np.ones((eparams.shape[-1]))
        else:
            cc = np.sign(pm)
        x = np.e**(-1j*(cc[:,None]*(freq[:,None]*(t[None,:] - mu[:,None])+p[:,None])))
        x*= y[:,None] * np.exp(-(t[None,:] - mu[:,None])**2 / (2*sigma[:,None]**2) ) 
        return t, x
    
    @classmethod
    def get_t(cls, d):
        return _get_t(cls,d)
