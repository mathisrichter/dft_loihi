import numpy as np
import nxsdk
import dft_loihi
import dft_loihi.dft.util


class Field(dft_loihi.dft.util.Connectable):
    """A population of neurons that acts as a dynamic neural field."""

    def __init__(
            self,
            name,
            net,
            sizes,
            kernel=None,
            tau_voltage=1,
            tau_current=1,
            threshold=100,
            with_probes=True,
            bias_mant=0,
            bias_exp=0):
        super().__init__()
        self.name = name
        self.threshold = threshold
        self.number_of_neurons = np.prod(sizes)  # multiplies the sizes of all dimensions

        compartment_prototype = nxsdk.net.nodes.compartments.CompartmentPrototype(
                              vThMant=self.threshold,
                              compartmentVoltageDecay=dft_loihi.dft.util.decay(tau_voltage),
                              compartmentCurrentDecay=dft_loihi.dft.util.decay(tau_current),
                              enableNoise=0,
                              refractoryDelay=1,
                              biasMant=bias_mant,
                              biasExp=bias_exp,
                              functionalState=nxsdk.api.enums.api_enums.COMPARTMENT_FUNCTIONAL_STATE.IDLE)

        self.neurons = net.createCompartmentGroup(size=self.number_of_neurons, prototype=compartment_prototype)
        self.input = self.neurons
        self.output = self.neurons
        
        #if (self_excitation > 0.001):
        #    dft_loihi.dft.util.connect(self, self, self_excitation, pattern="full")
#       #     connection_prototype_recurrent = nxsdk.net.nodes.connections.ConnectionPrototype(weight=self_excitation * self.threshold)
#       #     self.neurons.connect(self.neurons, prototype=connection_prototype_recurrent, connectionMask=np.ones((self.number_of_neurons, self.number_of_neurons)))
        
        if (with_probes):
            self.setup_probes()
 
    def setup_probes(self):
        (self.probe_current, self.probe_voltage, self.probe_spikes) = self.neurons.probe(
                 [nxsdk.api.enums.api_enums.ProbeParameter.COMPARTMENT_CURRENT,
                  nxsdk.api.enums.api_enums.ProbeParameter.COMPARTMENT_VOLTAGE,
                  nxsdk.api.enums.api_enums.ProbeParameter.SPIKE])

    def weight_transform(self, weight):
        return weight * self.threshold
