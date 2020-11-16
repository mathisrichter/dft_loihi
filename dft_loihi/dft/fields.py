import dft.util.decay

class Node:
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
        self.name = name
        self.number_of_neurons = number_of_neurons
        self.threshold = threshold
                
        compartment_prototype = nx.CompartmentPrototype(
                              vThMant=Node.THRESHOLD,                              
                              compartmentVoltageDecay=dft.util.decay(tau_voltage),
                              compartmentCurrentDecay=dft.util.decay(tau_current),
                              enableNoise=0,
                              refractoryDelay=1,
                              biasMant=biasMant,
                              biasExp=biasExp,
                              functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE)

        self.neurons = net.createCompartmentGroup(size=self.number_of_neurons, prototype=compartment_prototype)
        
        if (self_excitation / self.threshold > 0.001):
            connection_prototype_recurrent = nxsdk.net.nodes.connections.ConnectionPrototype(weight=self_excitation)
            self.neurons.connect(self.neurons, prototype=connection_prototype_recurrent, connectionMask=np.ones((self.number_of_neurons, self.number_of_neurons)))
        
        if (with_probes):
            setup_probes(self)
    
        self.visualization = None
    
    def setup_probes(self):
        (self.probe_current, self.probe_voltage, self.probe_spikes) = self.neurons.probe([nx.ProbeParameter.COMPARTMENT_CURRENT, nx.ProbeParameter.COMPARTMENT_VOLTAGE, nx.ProbeParameter.SPIKE])
