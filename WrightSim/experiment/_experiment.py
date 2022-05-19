# --- import --------------------------------------------------------------------------------------


import os

import numpy as np

import WrightTools as wt

from . import _pulse
from ._scan import Scan


# --- define --------------------------------------------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))


# integration defaults
timestep = 4.0
early_buffer = 100.0
late_buffer = 400.0


# --- class ---------------------------------------------------------------------------------------


class Experiment:
    """Experiment."""

    def __init__(self, axes, name, pm, pulse_class):
        # basic attributes
        self.axes = axes
        for a in self.axes:
            setattr(self, a.name, a)
        self.name = name
        self.pm = pm
        self.npulses = len(pm)
        self.timestep = timestep
        self.early_buffer = early_buffer
        self.late_buffer = late_buffer
        # pulse
        self.pulse_class = pulse_class
        self.pulses = [self.pulse_class() for _ in self.pm]

    def __repr__(self):
        return '<WrightSim.Experiment object \'{0}\' at {1}>'.format(self.name, str(id(self)))

    @property
    def active_axes(self):
        return [a for a in self.axes if a.active]

    @property
    def axis_names(self):
        return [a.name for a in self.axes]

    def run(self, hamiltonian, mp=True, windowed=True):
        """Run the experiment.

        Parameters
        ----------
        hamiltonian : WrightSim Hamiltonian
            Hamiltonian.
        mp : boolean (optional)
            Toggle CPU multiprocessing. Default is True.
        windowed : boolean (optional)
            Toggle truncating output bounds.  Default is True.
        Returns
        -------
        WrightSim Scan
            Scan that was run."""
        out = Scan(self, hamiltonian, windowed=windowed)
        out.run(mp=mp)
        # finish
        return out

    def set_axis(self, axis_name, points):
        '''
        Activate and define points for one of the experimental axes.

        Parameters
        ----------
        axis_name : string
            Name of axis.
        points : 1D array-like
            Points (in native units) to scan over.
        '''
        # TODO: is there a way to prevent incompatible axes being simultaniously activated?
        axis_index = self.axis_names.index(axis_name)
        axis = self.axes[axis_index]
        axis.points = points
        axis.active = True
