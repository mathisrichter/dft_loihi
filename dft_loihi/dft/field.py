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
            domain,
            shape,
            kernel=None,
            tau_voltage=1,
            tau_current=1,
            threshold=100,
            with_probes=True):
        super().__init__()
        self.name = name

        if type(shape) == int:
            domain = np.array([domain], dtype=np.float32)
            shape = (shape,)
        self.domain = domain
        self.shape = shape

        self.threshold = threshold
        self.number_of_neurons = np.prod(shape)  # multiplies the sizes of all dimensions

        compartment_prototype = nxsdk.net.nodes.compartments.CompartmentPrototype(
                              vThMant=self.threshold,
                              compartmentVoltageDecay=dft_loihi.dft.util.decay(tau_voltage),
                              compartmentCurrentDecay=dft_loihi.dft.util.decay(tau_current),
                              enableNoise=0,
                              refractoryDelay=1,
                              functionalState=nxsdk.api.enums.api_enums.COMPARTMENT_FUNCTIONAL_STATE.IDLE)

        self.neurons = net.createCompartmentGroup(size=self.number_of_neurons, prototype=compartment_prototype)
        self.input = self.neurons
        self.output = self.neurons

        if (kernel is not None):
            kernel.create(self.domain, self.shape)
            print("kernel: " + str(kernel.weights))
            dft_loihi.dft.util.connect(self, self, kernel.weights, mask=kernel.mask)

        if (with_probes):
            self.setup_probes()
 
    def setup_probes(self):
        (self.probe_current, self.probe_voltage, self.probe_spikes) = self.neurons.probe(
                 [nxsdk.api.enums.api_enums.ProbeParameter.COMPARTMENT_CURRENT,
                  nxsdk.api.enums.api_enums.ProbeParameter.COMPARTMENT_VOLTAGE,
                  nxsdk.api.enums.api_enums.ProbeParameter.SPIKE])

    def weight_transform(self, weight):
        return weight * self.threshold
