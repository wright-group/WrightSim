"""Axis."""


# --- import --------------------------------------------------------------------------------------


import numpy as np


# --- define --------------------------------------------------------------------------------------


__all__ = ['Axis']


# --- class ---------------------------------------------------------------------------------------


class Axis(object):

    def __init__(self, name, units=None, points=None, active=False, parameter=None, pulses=None,
                 cols=None):
        self.name = name
        self.units = units
        self.points = points
        self.active = active
        self.parameter = parameter
        self.pulses = pulses
        self.cols=cols
        print(self.cols)

    @property
    def coords(self):
        parameter_index = self.cols[self.parameter]
        self.coords = np.zeros((len(self.pulses), 2), dtype=np.int)
        for i, pulse in enumerate(self.pulses):
            # specify pulse and then efield param of pulse
            self.coords[i] = [pulse, parameter_index]
