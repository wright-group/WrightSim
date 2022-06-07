from WrightTools import _group as wt_group
import WrightTools as wt

from WrightSim import experiment

#from . import experiment

from  WrightSim.experiment import _scan as Scan
import numpy as np
import json
import pickle
import h5py
import pathlib
import os
import pandas as pd
from pandas import DataFrame as df
import time


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
    for k in range(len(wsexp.active_axes)):
        dataobj.create_variable(
            values=wsexp.active_axes[k].points,
            name=str(wsexp.active_axes[k].name),
            units=wsexp.active_axes[k].units,
            dtype = wsexp.active_axes[k].points.dtype 
            )
        strnglist.append(str(wsexp.active_axes[k].name))
    
    stringcut=str(strnglist)[1:-1]
    dataobj.create_channel(name="sim", values=wsscan.sig[:], dtype=np.complex128)
    eval(f"dataobj.transform({stringcut})")
    return dataobj


def save_exp(filepath, wsexp):
    """Save a pickled WrightSim experiment."""
    assert (type(wsexp)==type(experiment.builtin("trive")))
    strng=""
    def as_str(exp, strng=""):
        '''
        try:
            if exp.__dict__:
                for i in exp.__dict__.keys():
                    exp2=eval(f"exp.{i}")
                    strng2=as_str(exp2, strng)
                    strngo=strng2+strng
            else:
                for i in range(len(exp)):
                    exp2=exp[i]
                    strng2=as_str(exp2, strng)
                    strngo=strng2+strng
        except:
            strngo=strng+" "+eval(str(exp))+"\n"        
        return strngo
        '''
    h5py.File(filepath, "w")
    new = wt_group.Group(filepath=filepath, edit_local=True)
    # attrs
    for k, v in wsexp.attrs.items():
        new.attrs[k] = v
    # children
    #for k, v in wsexp.items():
    #    super().copy(v, new, name=v.natural_name)
    # finish
    new.flush()
    new.close()
    #strng2=as_str(wsexp)
    #for i in wsexp.__dict__.keys():
    #    print(i)
        #strng= strng+ f"{i}:" + f" {wsexp.{i}}\n"
    #out=pickle.dumps(wsexp)
    #unpickled=pd.read_pickle(out)
    #dfr=pd.DataFrame(dict(wsexp))
    #data=str(wsexp)
    #f=open(filename, "w")
    #f.write(strng)
    #f.close()
    return 


def load_exp(filename):
    """Load the JSON representation into an Experiment object."""
    f=open(filename, "r")
    jsonobj=json.load(f)
    f.close()
    
    #assert (type(exp)==type(experiment.builtin("trive")))
    return jsonobj


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


if __name__ == '__main__':
    exp = experiment.builtin('trive')
    save_exp("test2.json", exp)
    time.sleep(0.5)
    data=load_exp("test2.json")

    pass