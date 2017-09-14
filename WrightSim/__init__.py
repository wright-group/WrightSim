# --- import --------------------------------------------------------------------------------------


import os as _os
import sys as _sys
_dir = _os.path.dirname(__file__)

import configparser as _ConfigParser


# --- version information -------------------------------------------------------------------------


# read from VERSION file
_here = _os.path.abspath(_os.path.dirname(__file__))
with open(_os.path.join(_os.path.dirname(_here), 'VERSION')) as _version_file:
    __version__ = _version_file.read().strip()

# add git branch, if appropriate
_directory = _os.path.dirname(_os.path.dirname(__file__))
_p = _os.path.join(_directory, '.git', 'HEAD')
if _os.path.isfile(_p):
    with open(_p) as _f:
        __branch__ = _f.readline().rstrip().split(r'/')[-1]
    if __branch__ != 'master':
        __version__ += '-' + __branch__
else:
    __branch__ = None


# --- populate __all__ ----------------------------------------------------------------------------


__all__ = []

from . import experiments
from . import integration
from . import measure
from . import response
