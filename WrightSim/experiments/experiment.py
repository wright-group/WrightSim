### import ####################################################################


import os

import numpy as np

import WrightTools as wt

from . import pulse


### define ####################################################################


directory = os.path.dirname(__file__)

# integration defaults
timestep = 4.0
early_buffer = 100.0
late_buffer  = 400.0


### main class ################################################################


# experiments are the actual running of the data
class Experiment:

    def __init__(self, axes, name, pm, pulse_class_name):
        # basic attributes
        self.axes = axes
        for a in self.axes:
            setattr(self, a.name, a)
        self.name = name
        self.pm = pm
        self.npulses = len(pm)
        self.timestep = timestep
        self.early_buffer = early_buffer
        self.late_buffer  = late_buffer     
        # pulse
        self.pulse_class_name = pulse_class_name
        self.pulses = [pulse.__dict__[self.pulse_class_name]() for _ in self.pm]

    def __repr__(self):
        return 'WrightSim.experiments.experiment.Experiment object \'{0}\' at {1}'.format(self.name, str(id(self)))      
      
    @property
    def active_axes(self):
        return [a for a in self.axes if a.active]

    @property
    def axis_names(self):
        return [a.name for a in self.axes]
        
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
