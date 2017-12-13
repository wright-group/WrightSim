"""Axis."""


# --- import --------------------------------------------------------------------------------------


# --- define --------------------------------------------------------------------------------------


__all__ = ['Axis']


# --- class ---------------------------------------------------------------------------------------


class Axis(object):

    def __init__(self, name, units=None, points=None, active=False, parameter=None, pulses=None):
        self.name = name
        self.units = units
        self.points = points
        self.active = active
        self.parameter = parameter
        self.pulses = pulses
