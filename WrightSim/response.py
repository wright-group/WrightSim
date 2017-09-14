### import ####################################################################


import WrightTools as wt


### main class ################################################################


class Response(wt.data.Data):
    
    def __init__(self, axes, channels, constants=[], name='', source=None):
        wt.data.Data.__init__(self, axes=axes, channels=channels,
                              constants=constants, name=name, source=source)

    def save(self, p=None):
        # TODO: https://github.com/wright-group/WrightSim/issues/8
        raise NotImplementedError


### open from folder ##########################################################


def load(p):
    # TODO:
    raise NotImplementedError
