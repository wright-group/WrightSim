from WrightTools import _group as wt_group
import WrightTools as wt
from . import experiment as expt
import numpy as np
import json
import pickle
import h5py
import pathlib
import os

def to_wtdata(wsexp, wsrun):
    '''
    to_wtdata(wsexp, wsrun)
    -----------
    Converts a WrightSim experiment and run into a properly formatted WrightTools data object.

    Inputs
    -- 
    wsexp:  WrightSim experiment object.
    wsrun:  WrightSim run object.


    Output
    ----
    data:  WrightTools data object.

    '''
    return


def save_exp(filename, wsexp):
    """Save a pickled WrightSim experiment."""
    assert (isinstance(wsexp, expt.builtin('trive')))
    return pickle.dump(wsexp, filename)
    

def load_exp(filename):
    """Load the JSON representation into an Experiment object."""
    exp= pickle.load(filename)
    assert (isinstance(exp, expt.builtin('trive')))
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

    # copy to new file
    h5py.File(filepath, "w")
    new = wt_group(filepath=filepath, edit_local=True)
    # attrs
    for k, v in wsrun.items():
        new.items[k] = v
    # children
    #for k, v in wsrun.items():
    #    wt_group.copy(v, new, name=v.natural_name)
    # finish
    new.flush()
    new.close()
    del new
    return 


def load_run(filepath):
    '''
    Loads an hdf5 file into a WrightSim run object.
    
        Open any ws5 file, returning the top-level object (data or collection).

    Parameters
    ----------
    filepath : path-like
        Path to file.
        Can be either a local or remote file (http/ftp).
        Can be compressed with gz/bz2, decompression based on file name.

    Returns
    -------
    WrightSim Run object
        Root-level object in file.
    '''
    filepath = os.fspath(filepath)
    ds = np.DataSource(None)

    f = h5py.File(filepath, "r")
    class_name = f["/"].attrs["class"]
    name = f["/"].attrs["name"]
    f.close()
    if class_name == "Run":
        #obj = wt_data.Data(filepath=str(filepath), name=name, edit_local=True)
        pass
    else:
        obj = wt_group.Group(filepath=str(filepath), name=name, edit_local=True)

    return obj
