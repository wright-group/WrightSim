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
        self.cols = cols
        self._coords = None

    def __repr__(self):
        return "<WrightSim.Axis '{0}' at {1}>".format(self.name, id(self))

    @property
    def coords(self):
        if self._coords is not None:
            return self._coords
        parameter_index = self.cols.index(self.parameter)
        coords = np.zeros((len(self.pulses), 2), dtype=np.int)
        for i, pulse in enumerate(self.pulses):
            # specify pulse and then efield param of pulse
            coords[i] = [pulse, parameter_index]
        self._coords = coords
        return coords
