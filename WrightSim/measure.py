### import ####################################################################


import WrightTools as wt


### spectrum class ############################################################


class Spectrum:
    
    def __init__(self, axes, constants=[]):
        self.axes = axes
        self.constants = constants
        
    def measure(self, *args):
        # TODO: should return a data object or something?
        pass
    
    def save(self):
        # TODO:
        pass


### measurement tools #########################################################


# TODO: mono
# TODO: sld
