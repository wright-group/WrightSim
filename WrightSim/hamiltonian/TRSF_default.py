import numpy as np
from ..mixed import propagate


class Hamiltonian:
    
    
    def __init__(self, rho=None, tau=None, mu=None,
                 omega=None, 
                 labels = ['gg',
                           'Ig', 'ig',
                           '2I,g', '2i,g', 'c,g',
                           'ag', 'bg'
                           ],
                time_orderings = list(range(1,4))
                 ):
        if rho is None:
            self.rho = np.zeros(len(labels), dtype=np.complex64)
            self.rho[0] = 1.
        else:
            self.rho = rho
        
        if tau is None:
            self.tau = tau
        else:
            self.tau = np.array([np.inf, 
                                 1e3, 1e3, 
                                 5e2, 5e2, 5e2,
                                 10, 10])



    
    propagator = 'rk' 
    pc = False 
    # 8 elements
    dm_vector = ['g,g',
                 'I,g', 'i,g', 
                 '2I,g', '2i,g', 'c,g',
                 'a,g', 'b,g']
    out_group = [[6,7]]#[[4]]
    #--------------------------Oscillator Properties--------------------------
    rho_0 = np.zeros((len(dm_vector)), dtype=np.complex64)
    rho_0[0] = 1.
    # state central position
    wI = 1570.
    wi = 1590.
    wa = 19000.  # cm-1
    wb = 20500.  # cm-1

    # exciton-exciton coupling
    I_anharm = 20.0
    i_anharm  = 20.0
    Ii_coupling = 20.0  # cm-1

    # dephasing times, 1/fs
    G_Ig  = 1/1000
    G_ig  = 1/1000
    G_IIg = 1/500
    G_iig = 1/500
    G_cg  = 1/500
    G_ag  = 1/100
    G_bg  = 1/100
    
    #transition dipoles (a.u.)
    m_gI = 1.0
    m_gi = 1.0

    m_I2I = m_gI / np.sqrt(2)
    m_i2i = m_gi / np.sqrt(2)
    m_Ic  = m_gi
    m_ic  = m_gI
    
    m_2Ia = 1.0
    m_2Ib = 1.0
    m_2ia = 1.0
    m_2ib = 1.0
    m_ca  = 1.0
    m_cb  = 1.0
    
    m_ag = 1.0
    m_bg = 1.0
    
    #--------------------------Recorded attributes--------------------------
    out_vars = ['dm_vector', 'out_group', 'rho_0',
                'm_gI', 'm_gi', 
                'm_i2i', 'm_I2I', 'm_ic', 'm_Ic',
                'm_2Ia', 'm_2Ib', 'm_2ia', 'm_2ib', 'm_ca', 'm_cb', 
                'm_ag', 'm_bg',
                'wI', 'wi', 'wa', 'wb',
                'I_anharm', 'i_anharm',
                'Ii_coupling',
                'pc', 'propagator', 
                'G_Ig', 'G_ig', 
                'G_IIg', 'G_iig', 'G_cg',
                'G_ag', 'G_bg',
                ]

    #--------------------------Methods--------------------------

    def __init__(self, **kwargs):
        # inherit all class attributes unless kwargs has them; then use those 
        # values.  if kwargs is not an Omega attribute, it gets ignored
        # careful: don't redefine instance methods as class methods!
        for key, value in kwargs.items():
            if key in Omega.__dict__.keys(): 
                setattr(self, key, value)
            else:
                print('did not recognize attribute {0}.  No assignment made'.format(key))
        # with this set, initialize parameter vectors
        # w_0 is never actually used for computations; only for reporting back...
        self.w_0 = gen_w_0(self.wI, self.I_anharm,
                           self.wi, self.i_anharm,
                           self.Ii_coupling,
                           self.wa, self.wb
                           )
        self.Gamma = gen_Gamma_0(self.G_Ig, self.G_ig, 
                                 self.G_IIg, self.G_iig, self.G_cg, 
                                 self.G_ag, self.G_bg, 
                                 )

    def o(self, efields, t, wl):
        # combine the two pulse permutations to produce one output array
        E1, E2, E3 = efields[0:3]
        
        out1 = self._gen_matrix(E1, E2, E3, t, wl, E1first = False)
        out2 = self._gen_matrix(E1, E2, E3, t, wl, E1first = True)

        return np.array([out1, out2], dtype=np.complex64)
    
    # to get a dummy matrix that shows connectivities, run
    # _gen_matrix(1,1,1,0,np.zeros((len(dm_vector))))
    def _gen_matrix(self, E1, E2, E3, t, wl, E1first = True):
        wIg, wig, wIIg, wiig, wcg, wag, wbg = wl[1:]

        wII_I = wIIg - wIg
        wii_i = wiig - wig
        wc_I = wcg - wIg
        wc_i = wcg - wig

        wa_II = wag - wIIg
        wb_II = wbg - wIIg
        wa_ii = wag - wiig
        wb_ii = wbg - wiig
        wa_c = wag - wcg
        wb_c = wbg - wcg
        
        m_gI = self.m_gI
        m_gi = self.m_gi
    
        m_I2I = self.m_I2I
        m_i2i = self.m_i2i
        m_Ic  = self.m_Ic
        m_ic  = self.m_ic
        
        m_2Ia = self.m_2Ia
        m_2Ib = self.m_2Ib
        m_2ia = self.m_2ia
        m_2ib = self.m_2ib
        m_ca  = self.m_ca
        m_cb  = self.m_cb
        
        m_ag = self.m_ag
        m_bg = self.m_bg

        if E1first==True:
            first  = E1
            second = E2
        else:
            first  = E2
            second = E1
        O = np.zeros((len(t), len(wl), len(wl)), dtype=np.complex64)
        # from gg
        O[:,1,0] =  m_gI  * first  * rotor(-wIg*t)
        O[:,2,0] =  m_gi  * first  * rotor(-wig*t)
        # from Ig
        O[:,3,1] =  m_I2I  * second * rotor(-wII_I*t)
        O[:,5,1] =  m_ic   * second * rotor(-wc_I*t)
        # from ig
        O[:,4,2] =  m_i2i  * second * rotor(-wii_i*t)
        O[:,5,2] =  m_Ic   * second * rotor(-wc_i*t)
        # from IIg
        O[:,6,3] = m_2Ia * E3 * rotor(-wa_II *t) * m_ag
        O[:,7,3] = m_2Ib * E3 * rotor(-wb_II *t) * m_bg 
        # from iig
        O[:,6,4] = m_2ia * E3 * rotor(-wa_ii *t) * m_ag
        O[:,7,4] = m_2ib * E3 * rotor(-wb_ii *t) * m_bg
        # from cg
        O[:,6,5] = m_ca * E3 * rotor(-wa_c *t) * m_ag
        O[:,7,5] = m_cb * E3 * rotor(-wb_c *t) * m_bg

        # make complex according to Liouville Equation
        O *= complex(0,0.5)
        for i in range(O.shape[-1]):
            O[:,i,i] = -self.Gamma[i]

        return O

    def ws(self, inhom_object):
        return self.w_0


