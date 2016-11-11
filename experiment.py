### import ####################################################################


import os

import numpy as np

import WrightTools as wt

from . import pulse


### define ####################################################################


directory = os.path.dirname(__file__)

timestep = 4.0
early_buffer = 100.0
late_buffer  = 400.0


### main class ################################################################


# experiments are the actual running of the data
class Experiment:


    def __init__(self, pulse_class_name='GaussRWA', pm=[1, -1, 1]):
        """
        Things that could potentially be set as default (class attributes):
            timestep, buffers
        Things to be specified by the experiment (instance attributes):
            the pulses module used
            number of pulses
            conjugate pulses
            pulse "positions"
        """
        self.timestep = timestep
        self.early_buffer = early_buffer
        self.late_buffer  = late_buffer
        self.pm = pm
        self.npulses = len(pm)
        self.pulse_class_name = pulse_class_name
        # assign timestep values to the package
        # careful:  changing Experiment attributes will change the pulse defaults, too
        # if we want to change these bounds, what is the best way to do it?
        pulse_class = pulse.__dict__[pulse_class_name]
        # write time properties to the pulse class
        # do this now, or at the time of writing?
        pulse_class.timestep = self.__class__.timestep
        pulse_class.early_buffer = self.__class__.early_buffer
        pulse_class.late_buffer = self.__class__.late_buffer
        # extract properties from the pulse class--default positions are stored 
        # in pulse class
        defaults = pulse_class.defaults
        # initiate the pulse coordinates for all beams
        self.positions = np.tile(defaults, (self.npulses,1))
        # extract the lookup table
        self.cols = pulse_class.cols
        
    def set_coord(self, axis_object, pos):
        # set a default position for a laser beam
        for coord in axis_object.coords:
            self.positions[coord[0], coord[1]] = pos
            # figure out what pulse attribute this is
            # second coord index is cols lookup
            for key, value in self.cols.items():
                if value == coord[1]:
                    pulse_property = key
            pulse_number = coord[0]
            print('{0}{1} moved to {2}'.format(pulse_property, pulse_number, pos))

    def get_coords(self):
        # return the coordinates of all pulses
        for key, value in self.cols.items():
            print('{0}:  {1}'.format(key, self.positions[:,self.cols[key]]))


### maker methods #############################################################


def builtin(name):
    p = os.path.join(directory, 'experiments', name.lower() + '.ini')
    return from_ini(p)


def from_ini(p):
    ini = wt.kit.INI(p)
    print(ini.read('main', 'name'))
    # TODO:
