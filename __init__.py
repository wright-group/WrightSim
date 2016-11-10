### import ####################################################################


import os as _os
import sys as _sys
_dir = _os.path.dirname(__file__)

import configparser as _ConfigParser


### version information #######################################################


# get config
_ini_path = _os.path.join(_dir, 'main.ini')
if _sys.version[0] == '3':
    _config = _ConfigParser.ConfigParser()
else:
    _config = _ConfigParser.SafeConfigParser()
_config.read(_ini_path)

# attempt get git sha
try:
    _HEAD_file = _os.path.join(_dir, '.git', 'logs', 'HEAD')
    with open(_HEAD_file) as _f:
        for _line in _f.readlines():
            _sha = _line.split(' ')[1]  # most recent commit is last
except:
    _sha = '0000000'

# version format: a.b.c.d
# a - major release
# b - minor release
# c - bugfix
# d - git sha key
__version__ = _config.get('main', 'version') + '.' + _sha[:7]


### populate __all__ ##########################################################


#from . import lib, hamiltonians, experiments

__all__ = []
#__all__.extend(lib.__all__)
#__all__.extend(hamiltonians.__all__)
