import numpy as np
import nxsdk
import dft_loihi
import dft_loihi.dft.util

class Node(dft_loihi.dft.util.Connectable):
    """A population of neurons that acts as a single dynamical node."""

    def __init__(
            self,
            name,
            net,
            number_of_neurons=1,
            tau_voltage=1,
            tau_current=1,
            threshold=100,
            self_excitation=0.0,
            with_probes=True,
            biasMant=0,
            biasExp=0):
        """Constructor
        
        Parameters:
        name --- A string that describes the node.
        net --- The nxsdk network object required to create the neurons.
        number_of_neurons --- Number of neurons to use for a single node.
        self_excitation --- Synaptic weight connecting the node to itself.
        with_probes --- Setting this flag to True will set up probes for the current, voltage, and spikes of all neurons.
        """
        super().__init__()
        self.name = name
        self.number_of_neurons = number_of_neurons
        self.threshold = threshold
                
        compartment_prototype = nxsdk.net.nodes.compartments.CompartmentPrototype(
                              vThMant=self.threshold,
                              compartmentVoltageDecay=dft_loihi.dft.util.decay(tau_voltage),
                              compartmentCurrentDecay=dft_loihi.dft.util.decay(tau_current),
                              enableNoise=0,
                              refractoryDelay=1,
                              biasMant=biasMant,
                              biasExp=biasExp,
                              functionalState=nxsdk.api.enums.api_enums.COMPARTMENT_FUNCTIONAL_STATE.IDLE)

        self.neurons = net.createCompartmentGroup(size=self.number_of_neurons, prototype=compartment_prototype)
        self.input = self.neurons
        self.output = self.neurons
        
        if (self_excitation > 0.001):
            dft_loihi.dft.util.connect(self, self, self_excitation, mask="full")
#            connection_prototype_recurrent = nxsdk.net.nodes.connections.ConnectionPrototype(weight=self_excitation * self.threshold)
#            self.neurons.connect(self.neurons, prototype=connection_prototype_recurrent, connectionMask=np.ones((self.number_of_neurons, self.number_of_neurons)))
        
        if (with_probes):
            self.setup_probes()
    
        self.visualization = None
 
    def setup_probes(self):
        (self.probe_current, self.probe_voltage, self.probe_spikes) = self.neurons.probe(
                 [nxsdk.api.enums.api_enums.ProbeParameter.COMPARTMENT_CURRENT,
                  nxsdk.api.enums.api_enums.ProbeParameter.COMPARTMENT_VOLTAGE,
                  nxsdk.api.enums.api_enums.ProbeParameter.SPIKE])

    def weight_transform(self, weight):
        return weight * self.threshold
