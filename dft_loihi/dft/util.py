import numpy as np
import nxsdk

from dft_loihi import dft, inputs


def decay(tau):
    return int(4095 / tau)

class Connectable:
    def __init__(self):
        self.input = None
        self.output = None

    def weight_transform(self, weight):
        return weight

def connect(source, target, weight):

    if isinstance(source, nxsdk.net.groups.CompartmentGroup):
        source_output = source
    else:
        source_output = source.output

    synaptic_weight = weight
    if isinstance(target, nxsdk.net.groups.CompartmentGroup):
        target_input = target
    else:
        target_input = target.input
        synaptic_weight = target.weight_transform(weight)

    prototype = nxsdk.net.nodes.connections.ConnectionPrototype(weight=synaptic_weight)
    source_output.connect(target_input, prototype=prototype)
