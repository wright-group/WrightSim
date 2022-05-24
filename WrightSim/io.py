from WrightTools import _group as wt_group
import WrightTools as wt

from . import experiment
from  .experiment import _scan as Scan
import numpy as np
import json
import pickle
import h5py
import pathlib
import os


def to_wtdata(wsexp, wsscan):
    '''
      
    to_wtdata(wsexp, wsrun)
    -----------
    Converts a WrightSim experiment and run into a properly formatted WrightTools data object.

    Inputs
    -- 
    wsexp:  WrightSim experiment object.
    wsscan:  WrightSim Scan object.


    Output
    ----
    data:  WrightTools data object.

    '''
    dataobj=wt.data.Data()
    strnglist=list()
    for k in wsscan.active_axes:
        dataobj.create_variable(
            values=wsscan.active_axes[k].points,
            name=str(wsscan.active_axes[k].name),
            units=wsscan.active_axes[k].units,
            dtype = wsscan.active_axes[k].points.type 
            )
        strnglist.append(str(wsscan.active_axes[k].name))
    
    stringcut=strnglist[1:-1]
    dataobj.create_channel(name="sim", values=wsscan.sig[:], dtype=np.complex128)
    dataobj.transform(stringcut)
    return dataobj


def save_exp(filename, wsexp):
    """Save a pickled WrightSim experiment."""
    assert (type(wsexp)==type(experiment.builtin("trive")))
    pickleb=pickle.dumps(wsexp)
    f=open(filename, "wb")
    f.write(pickleb)
    f.close()
    return 


def load_exp(filename):
    """Load the JSON representation into an Experiment object."""
    f=open(filename, "rb")
    exp= pickle.load(f)
    f.close()
    assert (type(exp)==type(experiment.builtin("trive")))
    return exp


def save_run(filepath, wsrun):
    '''
    Saves a WrightSim run result to an hdf5 file.
   

    Save as root of a new file.

        Parameters
        ----------
        filepath : Path-like object (optional)
            Filepath to write.
        wsrun :  WrightSim run object

        '''
    filepath = pathlib.Path(filepath)
    filepath = filepath.with_suffix(".ws5")
    filepath = filepath.absolute().expanduser()

  
    f=h5py.File(filepath, "w")
    dat=f.create_group("WrightSim")
    dat.create_dataset("sig", data=wsrun.sig[:])

    f.flush()
    f.close()

    return 


def load_run(filepath):
    '''
    Loads an hdf5 file into a Scan.sig array.
    
        Open any ws5 file, returning the top-level object (data or collection).

    Parameters
    ----------
    filepath : path-like
        Path to file.

    Returns
    -------
    WrightSim Scan.sig array
        
    '''
    filepath = os.fspath(filepath)

    f = h5py.File(filepath, "r")
    
    
    obj = np.array(f['WrightSim/sig'][:])
    f.flush()
    f.close()

    return obj
