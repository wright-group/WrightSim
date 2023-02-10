### import ####################################################################
import WrightTools as wt
import numpy as np


def convert(scan, polarizations=True, mulist=None):
    ''' convert a WrightSim scan.sig output into a WrightTools data object.
    
    Parameters:
    
    scan -- WrightSim scan object populated with scan.sig from a run
    
    polarizations -- if set to True, converts density matrix elements to polarizations by multiplying by mus
    in the mulist and summing with the product's complex conjugate. Does not multiply by number density or by
    local field factors.

    mulist (np.complex128) -- 1D list of mus with values similar to WrightSim experiment.hamiltonian.mu,
                                with length equal to the number of density_matrix elements being converted to emitting dipoles

    '''    
       
    units_dict= {
        "A" : None,
        "s" : "fs",
        "d" : "fs",
        "w" : "wn",
        "p" : "rad"
    }
    try:    
        scan.sig
    except:
        print ("no scan.sig")   
        return 0
    finally:
        data=wt.data.Data()
        for axis_obj in scan.axis_objs:
            axis_obj_units=units_dict[axis_obj.parameter]    
            data.create_variable(name=axis_obj.name, values=axis_obj.points, units=axis_obj_units)
        data.create_variable(name="out", values=scan.t, units="fs")
        sig_swapped=scan.sig.swapaxes(-2,-1)
        if polarizations:
            if mulist==None:
                mulist=np.ones(len(scan.ham.recorded_indices),dtype=np.complex128)
            for idx,rec_idx in enumerate(scan.ham.recorded_indices):
                val=np.real(sig_swapped[...,idx]*mulist[idx]+np.conjugate(sig_swapped[...,idx]*mulist[idx]))
                data.create_channel(name='P '+scan.ham.labels[rec_idx], values=val, dtype=np.float64)
        else:
            for idx,rec_idx in enumerate(scan.ham.recorded_indices):
                data.create_channel(name='rho '+scan.ham.labels[rec_idx], values=sig_swapped, dtype=np.complex128)
        return data


