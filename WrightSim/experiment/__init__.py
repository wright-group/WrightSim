# --- import --------------------------------------------------------------------------------------


import os

import WrightTools as wt

from . import _pulse
from ._experiment import Experiment
from ._axis import Axis


# --- define --------------------------------------------------------------------------------------


directory = os.path.dirname(__file__)


# --- getters -------------------------------------------------------------------------------------


def builtin(name):
    p = os.path.join(directory, name.lower() + '.ini')
    return from_ini(p)


def from_ini(p):
    ini = wt.kit.INI(p)
    # pulse
    pulse_class_name = ini.read('main', 'pulse class name')
    pulse_class = _pulse.__dict__[pulse_class_name]
    # axes
    axis_names = ini.sections
    axis_names.remove('main')
    axes = []
    for name in axis_names:
        is_axis = ini.read(name, 'axis')
        if is_axis: 
            points = ini.read(name, 'points')
            active = ini.read(name, 'active')
            pulses = ini.read(name, 'pulses')
            parameter = ini.read(name, 'parameter')
            axis = Axis(points=points, units=None, name=name, active=active, pulses=pulses,
                    parameter=parameter, cols=pulse_class.cols)
            axes.append(axis)
    # construct experiment object
    name = ini.read('main', 'name')
    pm = ini.read('main', 'pm')
    e = Experiment(axes=axes, name=name, pm=pm, pulse_class=pulse_class)
    # finish
    return e
