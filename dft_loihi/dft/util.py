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

def connect(source, target, weight, pattern="one-to-one"):

    if isinstance(source, nxsdk.net.groups.CompartmentGroup):
        source_output = source
        source_num_neurons = source.size
    else:
        source_output = source.output
        source_num_neurons = source.number_of_neurons

    synaptic_weight = weight
    if isinstance(target, nxsdk.net.groups.CompartmentGroup):
        target_input = target
        target_num_neurons = target.size
    else:
        target_input = target.input
        target_num_neurons = target.number_of_neurons
        synaptic_weight = target.weight_transform(weight)

    if (pattern == "full"):
        mask = np.full((source_num_neurons, target_num_neurons), 1.0)
    elif (pattern == "one-to-one"):
        mask = np.eye(source_num_neurons, target_num_neurons)

    prototype = nxsdk.net.nodes.connections.ConnectionPrototype(weight=synaptic_weight)
    source_output.connect(target_input, prototype=prototype, connectionMask=mask)
