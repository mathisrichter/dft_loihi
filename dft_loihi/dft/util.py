def decay(tau):
    return int(4095 / tau)

def connect(source, target, synaptic_weight=1.1 * Node.THRESHOLD):
    """Connects a node (or an input) to another node."""

    source_population = None
    mark = None
    
    if isinstance(source, Node):
        source_population = source.neurons
        mask = np.full((source.number_of_neurons, target.number_of_neurons), 1.0)
    elif isinstance(source, Input):
        source_population = source.input
        mask = np.eye(source.number_of_neurons)

    prototype = nxsdk.net.nodes.connections.ConnectionPrototype(weight=synaptic_weight)
        
    if isinstance(target, Node):
        source_population.connect(target.neurons,
                                  prototype=prototype,
                                  connectionMask=mask)
    else:
        source_population.connect(target,
                                 prototype=prototype)
