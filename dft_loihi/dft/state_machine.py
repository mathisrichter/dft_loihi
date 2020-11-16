# Mathis Richter (mathis.richter@ini.rub.de), 2020

import math
import matplotlib.pyplot as plt
import numpy as np
import nxsdk.api.n2a as nx
import nxsdk.net.groups
from nxsdk.utils.plotutils import plotRaster
from time import sleep


# for visualization
import matplotlib.animation
from matplotlib import rc
from IPython.display import HTML, Image, display
rc('animation', html='html5')
#matplotlib.pyplot.rcParams['animation.convert_path'] = '/homes/mathis.richter/programs/imagemagick_install/bin/magick'


def decay(tau):
    return int(4095 / tau)

class Node:
    """A population of neurons that acts as a single dynamical node."""
    
    class Visualization(object):
        def __init__(self, x=0, y=0, active=False, radius=0.5):
            self.radius = 0.5
            self.x = x
            self.y = y
            self.circle = None
            
        def draw(self):
            self.circle = matplotlib.pyplot.Circle((self.x,self.y),
                                              radius=self.radius,
                                              fill=True,
                                              ec="k",
                                              fc="w",
                                              zorder=2)
            matplotlib.pyplot.gca().add_patch(self.circle)
        
        def update(self, active):
            fillcolor = "dodgerblue" if (active) else "w"
            self.circle.set_fc(fillcolor)
            
    class PreconditionVisualization(Visualization):
        def __init__(self):
            self.sources = {}
            self.target = ""
            self.task_node = None
            
            Node.Visualization.__init__(self)
            
        def add_source_cos(self, behavior_name, cos_name):
            self.sources[behavior_name] = cos_name
            
        def set_target_behavior(self, target_name):
            self.target = target_name
    
    # these default parameters are used throughout for all nodes
    TAU_VOLTAGE = 1
    TAU_CURRENT = 1
    THRESHOLD = 100
    
    def __init__(self, name, net, number_of_neurons, self_excitation=0.0, with_probes=True, biasMant=0, biasExp=0):
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
                
        compartment_prototype = nx.CompartmentPrototype(
                              vThMant=Node.THRESHOLD,                              
                              compartmentVoltageDecay=decay(Node.TAU_VOLTAGE),
                              compartmentCurrentDecay=decay(Node.TAU_CURRENT),
                              enableNoise=0,
                              refractoryDelay=1,
                              biasMant=biasMant,
                              biasExp=biasExp,
                              functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE)

        self.neurons = net.createCompartmentGroup(size=self.number_of_neurons, prototype=compartment_prototype)
        
        if (self_excitation / Node.THRESHOLD > 0.001):
            connection_prototype_recurrent = nxsdk.net.nodes.connections.ConnectionPrototype(weight=self_excitation)
            self.neurons.connect(self.neurons, prototype=connection_prototype_recurrent, connectionMask=np.ones((self.number_of_neurons, self.number_of_neurons)))
        
        # set up probes for current, voltage and spikes
        if (with_probes):
            (self.probe_current, self.probe_voltage, self.probe_spikes) = self.neurons.probe([nx.ProbeParameter.COMPARTMENT_CURRENT, nx.ProbeParameter.COMPARTMENT_VOLTAGE, nx.ProbeParameter.SPIKE])
    
        self.visualization = None
    
    def plot(self):
        """Creates plots for all probes of the node."""
        if (self.probe_current == None or self.probe_voltage == None or self.probe_spikes == None):
            print("Error: Node does not have all necessary probes to plot.")
        else:
            fig = plt.figure(figsize=(18,10))
            fig.suptitle(self.name)

            ax0 = plt.subplot(3,1,1)
            self.probe_current.plot()
            plt.xlabel('Time steps')
            plt.ylabel('Current')
            plt.title('Current')

            ax1 = plt.subplot(3,1,2)
            self.probe_voltage.plot()
            plt.xlabel('Time steps')
            plt.ylabel('Voltage')
            plt.title('Voltage')

            ax2 = plt.subplot(3,1,3)
            self.probe_spikes.plot()
            plt.xlabel('Time steps')
            plt.ylabel('Neuron index')
            plt.title('Spikes')
            
            ax1.set_xlim(ax0.get_xlim())
            ax2.set_xlim(ax0.get_xlim())
            
            plt.tight_layout()
            plt.show()

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
    

class Behavior:
    
    class Visualization():
        def __init__(self):
            self.subbehaviors = []
    
    """A small network of dynamic neural nodes that represent and control a cognitive operation (or behavior)."""
    def __init__(self, name, net, neurons_per_node, has_subbehaviors=False):
        """Constructor
        
        Parameters:
        name --- A string that describes the behavior.
        net --- The nxsdk network object required to create the neurons.
        neurons_per_node --- Number of neurons to use for each node within the behavior.
        has_subbehaviors --- Set this flag to true if this behavior does not do any direct work but only connects to a set of subbehaviors.
        """
        self.name = name
        self.net = net
        self.neurons_per_node = neurons_per_node
        self.has_subbehaviors = has_subbehaviors
        
        # activated and sustained by input from task (or higher-level intention node), turns off without that input
        self.node_prior_intention = Node("Prior Intention", net, neurons_per_node)
        # activated by input from prior intention node, turns off without that input
        self.node_intention = Node("Intention", net, neurons_per_node)
        
        # a dictionary for CoS memory nodes
        self.nodes_cos_memory = {}
        # a dictionary for CoS nodes
        self.nodes_cos = {}
        # a dictionary for precondition nodes
        self.preconditions = {}

        # the prior intention node activates the intention node
        connect(self.node_prior_intention, self.node_intention, synaptic_weight=1.1 * Node.THRESHOLD)
        
        # if this behavior only has subbehaviors, add a single CoS signal
        # it will later be fed from the CoS signals of all subbehaviors
        if (self.has_subbehaviors):
            single_cos_name = "done"
            self.add_cos(single_cos_name)
            self.single_node_cos = self.nodes_cos[single_cos_name]
        
        self.visualization = Behavior.Visualization()
    
    def register(self, groups, behavior_dictionary, output=None, input=None):
        behavior_dictionary[self.name] = self
        
        groups[self.name + ".prior_intention"] = self.node_prior_intention.neurons
        groups[self.name + ".intention"] = self.node_intention.neurons
        
        if (output != None):
            output[self.name + ".start"] = self.node_intention.neurons
            output[self.name + ".prior_intention"] = self.node_prior_intention.neurons # for debugging
        
        for key in self.nodes_cos.keys():
            groups[self.name + ".cos." + key] = self.nodes_cos[key].neurons
            groups[self.name + ".cos_memory." + key] = self.nodes_cos_memory[key].neurons
            
            if (input != None):
                input[self.name + "." + key] = self.nodes_cos[key].neurons
                
            # for debugging
            if (output != None):
                output[self.name + ".cos." + key] = self.nodes_cos[key].neurons
                output[self.name + ".cos_memory." + key] = self.nodes_cos_memory[key].neurons
                
    def register_probes(self, probes):
        probes[self.name + ".prior_intention"] = self.node_prior_intention.probe_spikes
        probes[self.name + ".intention"] = self.node_intention.probe_spikes
        
        for name,node in self.nodes_cos.items():
            probes[self.name + ".cos." + name] = node.probe_spikes
        
        for name,node in self.nodes_cos_memory.items():
            probes[self.name + ".cos_memory." + name] = node.probe_spikes
            
        for name,node in self.preconditions.items():
            probes[self.name + ".precondition." + name] = node.probe_spikes
    
    def add_cos(self, cos_name, cos_input=None):
        """Adds a condition-of-satisfaction (CoS) to the behavior.
        
        Parameters:
        name --- A string that describes the CoS.
        cos_input --- The input that provides the CoS signal. Leave empty if you want to add it manually later.
        """
        # activated by input from BOTH intention node and "sensory" input, turns off without either
        cos_node = Node(cos_name, self.net, self.neurons_per_node)
        self.nodes_cos[cos_name] = cos_node
        # activated by input from BOTH task and cos node, remains on without cos input, turns off without task input
        cos_memory_node = Node(cos_name + " Memory", self.net, self.neurons_per_node, self_excitation=0.3 * Node.THRESHOLD)
        self.nodes_cos_memory[cos_name] = cos_memory_node
        
        int_cos_weight = 0.6 if (not self.has_subbehaviors) else 1
        connect(self.node_intention, cos_node, synaptic_weight=int_cos_weight * Node.THRESHOLD)
        connect(cos_node, cos_memory_node, synaptic_weight=0.3 * Node.THRESHOLD)
        connect(cos_memory_node, self.node_intention, synaptic_weight=-1.0 * Node.THRESHOLD)
        
        if (cos_input != None):
            connect(cos_input, cos_node, synaptic_weight=0.6 * Node.THRESHOLD)
        
    def add_boost(self, boost_input):
        """Adds an artificial boost input to the behavior, which may activate it; relevant for top-level behaviors."""
        connect(boost_input, self.node_prior_intention, synaptic_weight=1.1 * Node.THRESHOLD)
        for cos_memory in self.nodes_cos_memory.values():
            connect(boost_input, cos_memory, synaptic_weight=0.8 * Node.THRESHOLD)
        
    def add_subbehaviors(self, sub_behaviors, logical_or=False):
        """Connects a list of other behaviors in such a way that they are on the next lower hierarchical level
        with respect to the current behavior.
        
        Parameters:
        sub_behaviors --- A list of subbehaviors. By default, all their CoS signals have to be achieved
                          in order to trigger this behavior's CoS.
        logical_or --- By setting this flag to true, this behavior requires only one of the subbehaviors' CoS to finish.
        """
        for sub_behavior in sub_behaviors:
            connect(self.node_intention, sub_behavior.node_prior_intention, synaptic_weight=1.1 * Node.THRESHOLD)
            
            weight = -1.1
            if logical_or:
                weight = weight / len(sub_behaviors)
            connect(sub_behavior.node_prior_intention, self.single_node_cos, synaptic_weight=weight * Node.THRESHOLD)
            
            for sub_cos_memory in sub_behavior.nodes_cos_memory.values():
                connect(self.node_intention, sub_cos_memory, synaptic_weight=0.8 * Node.THRESHOLD)
                connect(sub_cos_memory, self.single_node_cos, synaptic_weight=1.2 * Node.THRESHOLD)
                
            self.visualization.subbehaviors.append(sub_behavior.name)
        
    def add_precondition(self, name, precondition_behaviors, cos_names, task_input, logical_or=False, register_groups=None):
        """Connects a list of other behaviors in such a way that they are on the next lower hierarchical level
        with respect to the current behavior.
        
        Parameters:
        name --- A string description of the precondition.
        precondition_behaviors --- A list of behaviors whose successful execution are a precondition to start executing this behavior.
        By default, all precondition behaviors have to be successfully executed to start executing this behavior.
        cos_names --- A list of lists of CoS names, e.g., [["CoS 1, first behavior", "CoS 2, first behavior"], ["CoS 1, second behavior"]]
        task_input --- Input activating the precondition nodes.
        logical_or --- By setting this flag to true, this behavior requires only one of the preconditions to start execution.
        """
        precondition_node = Node(name, self.net, self.neurons_per_node)
        self.preconditions[name] = precondition_node
        if (register_groups != None):
            register_groups[name] = precondition_node.neurons
        connect(precondition_node, self.node_intention, synaptic_weight=-1.1 * Node.THRESHOLD)
        precondition_node.visualization = Node.PreconditionVisualization()
        precondition_node.visualization.set_target_behavior(self.name)
        precondition_node.visualization.task_node = task_input
        for i in range(len(precondition_behaviors)):
            for j in range(len(cos_names[i])):
                weight = -1.1
                if (not logical_or):
                    weight = weight / len(precondition_behaviors)
                connect(precondition_behaviors[i].nodes_cos_memory[cos_names[i][j]],
                        precondition_node,
                        synaptic_weight=weight * Node.THRESHOLD)
                precondition_node.visualization.add_source_cos(precondition_behaviors[i].name, cos_names[i][j])
        connect(task_input, precondition_node, synaptic_weight=1.1 * Node.THRESHOLD)
        
    def plot(self):
        """Sets up plots for all nodes of the behavior."""
        print(self.name)
        self.node_prior_intention.plot()
        self.node_intention.plot()
        for name in self.nodes_cos:
            self.nodes_cos[name].plot()
            self.nodes_cos_memory[name].plot()
            
class Input:
    """
    An artificial spike pattern that can be used as an input to nodes.
    The pattern can be shaped by specifying phases of time in which all neurons have a certain probability of spiking at every time step.
    
    After creating the input object, add a number of input phases (add_input_phase function) and then create the overall input time couse (create_input() function.
    """
    
    def __init__(self, name, net, number_of_neurons, number_of_timesteps):
        """Constructor
        
        Parameters:
        name --- A string that describes the input.
        net --- The nxsdk network object required to create the neurons.
        number_of_neurons --- Number of neurons for this input.
        number_of_timesteps --- The total number of time steps over which the input time course will be generated."""
        self.name = name
        self.number_of_neurons = number_of_neurons
        self.number_of_timesteps = number_of_timesteps
        self.spike_times = []
        self.input_phases = []
        
        self.input = net.createSpikeGenProcess(numPorts=self.number_of_neurons)
        
    def add_input_phase_probability(self, length, spiking_probability):
        """Adds a phase with a given length in time steps, during which each neuron has a given probability to spike at every time step.
        
        Parameters:
        length --- The length of the phase in time steps.
        spiking_probability --- The probability of a neuron to spike at any tim step within the phase [0,1]. Same for all neurons."""
        spikes = np.random.rand(self.number_of_neurons, length) < spiking_probability
        self.input_phases.append(spikes)
    
    def add_input_phase_spike_distance(self, length, spike_distance):
        """Adds a phase with a given length in time steps, during which each neuron spikes at regular intervales of spike_distance.
        
        Parameters:
            length: The length of the phase in time steps.
            spike_distance: Number of time steps between each spike."""
        spikes = np.zeros((self.number_of_neurons, length))
        spikes[:, 0::spike_distance] = 1
        self.input_phases.append(spikes)
        
    def add_input_phase_on_off(self, length, on):
        spikes = np.zeros((self.number_of_neurons, length))
        if (on):
            spikes += 1
        self.input_phases.append(spikes)
        
    def create_empty_input(self, length):
        self.input_phases.append(np.zeros((self.number_of_neurons, length)))
        self.create_input()
        
    def create_complete_input(self, length, start_on, length_on):
        self.add_input_phase_on_off(start_on-1, on=False)
        self.add_input_phase_on_off(length_on, on=True)
        self.add_input_phase_on_off(length-(start_on-1)-length_on, on=False)
        self.create_input()
        
    def create_input(self):
        """Creates the overall time course of the input.
        Make sure to only call this once you have added all phases of input with the add_input_phase() function."""
        
        concatenated_phases = np.empty((self.number_of_neurons, 0))
        
        for phase in self.input_phases:
            concatenated_phases = np.append(concatenated_phases, phase, axis=1)
                
        for i in range(self.number_of_neurons):
            st = np.where(concatenated_phases[i,:])[0].tolist()
            self.spike_times.append(st)
            self.input.addSpikes(spikeInputPortNodeIds=i, spikeTimes=st)
        
    def plot(self):
        """Create a plot of the spike pattern of the input over time."""
        fig = plt.figure(figsize=(18,3))
        fig.suptitle(self.name)
        plotRaster(self.spike_times)
        plt.xlabel('Time steps')
        plt.ylabel('Input index')
        plt.xlim(0, self.number_of_timesteps)
        plt.title('Source Spikes')
        plt.tight_layout()
        plt.show()
    

class StateMachine():
        
    def __init__(self, net):
        
        query_behavior = False

        # parameters
        self.NEURONS_PER_NODE = 1

        self.net = net

        # return values

        self.probed_groups = {}   # a dictionary of all compartments that produce output for YARP
        self.out_groups = {}   # a dictionary of all compartments that produce output to other networks on Loihi

        self.input_from_yarp = {}  # a dictionary of all compartments that receive input from YARP
        self.input_from_loihi = {}  # a dictionary of all compartments that receive input from other networks on Loihi

        self.behaviors = [None] * 7 # a list of all behaviors (in the order in which they should be plotted)
        self.behavior_dictionary = {}
        
        self.groups = {} # a dictionary of all the compartment groups

        behavior_look = self.create_behavior(
            "look",
            plot_index=0,
            output=self.probed_groups,
            has_subbehaviors=True)

        behavior_look_at_object = self.create_behavior(
            "state_machine.look_at_object",
            plot_index=1,
            cos_names=["done"],
            output=self.probed_groups,
            input=self.input_from_yarp)

        behavior_recognize_object = self.create_behavior(
            "state_machine.recognize_object",
            plot_index=2,
            cos_names=["known", "unknown"],
            output=self.probed_groups,
            input=self.input_from_loihi)

        behavior_learn_new_object = self.create_behavior(
            "state_machine.learn_new_object",
            plot_index=3,
            cos_names=["done"],
            output=self.probed_groups,
            input=self.input_from_yarp)
        
        behavior_dummy = self.create_behavior(
            "state_machine.dummy",
            plot_index=4,
            cos_names=["done"],
            output=self.probed_groups,
            input=self.input_from_yarp)
        # make this dummy behavior instantly create its own CoS signal
        connect(behavior_dummy.node_intention,
                behavior_dummy.nodes_cos["done"],
                synaptic_weight=1.1 * Node.THRESHOLD)

    
        behavior_look.add_subbehaviors([behavior_look_at_object, behavior_recognize_object])
        behavior_look.add_subbehaviors([
            behavior_learn_new_object,
            behavior_dummy],
            logical_or=True)
        
        behavior_recognize_object.add_precondition(
            "prec.look_at_object:recognize_object",
            [behavior_look_at_object],
            [["done"]],
            behavior_look.node_intention,
            register_groups=self.groups)
        
        self.probed_groups["state_machine.recognize_object.prec.look_at_object"] = behavior_recognize_object.preconditions["prec.look_at_object:recognize_object"]
        
        behavior_dummy.add_precondition(
            "prec.recognize_object:dummy",
            [behavior_recognize_object],
            [["known"]],
            behavior_look.node_intention,
            register_groups=self.groups)
        
        self.probed_groups["state_machine.dummy.prec.recognize_object"] = behavior_dummy.preconditions["prec.recognize_object:dummy"]

        behavior_learn_new_object.add_precondition(
            "prec.recognize_object:learn_new_object",
            [behavior_recognize_object],
            [["unknown"]],
            behavior_look.node_intention,
            register_groups=self.groups)
        
        self.probed_groups["state_machine.learn_new_object.prec.recognize_object"] = behavior_learn_new_object.preconditions["prec.recognize_object:learn_new_object"]



        behavior_query = self.create_behavior(
            "query",
            plot_index=5,
            output=self.probed_groups,
            has_subbehaviors=True)

        behavior_query_memory = self.create_behavior(
            "state_machine.query_memory",
            plot_index=6,
            cos_names=["done"],
            output=self.probed_groups,
            input=self.input_from_yarp)

        behavior_query.add_subbehaviors([behavior_query_memory])
        

        look_bias = 100
        query_bias = 0
        if (query_behavior):
            look_bias = 0
            query_bias = 100
            
        # top node that activates everything if it receives input
        node_look_name = "Node look"
        node_look = Node(node_look_name, net, self.NEURONS_PER_NODE, biasMant=look_bias, biasExp=7)
        self.groups[node_look_name] = node_look
        behavior_look.add_boost(node_look)
        connect(behavior_look.nodes_cos_memory["done"], node_look, synaptic_weight=-1.5 * Node.THRESHOLD)

        node_query_name = "Node query"
        node_query = Node(node_query_name, net, self.NEURONS_PER_NODE, biasMant=query_bias, biasExp=7)
        self.groups[node_query_name] = node_query
        behavior_query.add_boost(node_query)
        connect(behavior_query.nodes_cos_memory["done"], node_query, synaptic_weight=-1.5 * Node.THRESHOLD)
        
        # for visualization
        self.probes = {}
        for behavior in self.behaviors:
            if (not behavior.has_subbehaviors):
                behavior.register_probes(self.probes)
    
    def create_behavior(self, name, plot_index, cos_names=None, has_subbehaviors=False, output=None, input=None):
        behavior = Behavior(name, self.net, self.NEURONS_PER_NODE, has_subbehaviors=has_subbehaviors)
        self.behaviors[plot_index] = behavior
        
        if (cos_names != None):
            for cos_name in cos_names:
                behavior.add_cos(cos_name)
        
        behavior.register(self.groups, self.behavior_dictionary, output=output, input=input)
        
        return behavior
    
    def connect_in(self, out_group):
        cos_weight = 0.6 * Node.THRESHOLD

        connect(out_group["software.look_at_object.done"], self.input_from_yarp["state_machine.look_at_object.done"], synaptic_weight=cos_weight)
        connect(out_group["object_recognition.recognize_object.known"], self.input_from_loihi["state_machine.recognize_object.known"], synaptic_weight=cos_weight)
        connect(out_group["object_recognition.recognize_object.unknown"], self.input_from_loihi["state_machine.recognize_object.unknown"], synaptic_weight=cos_weight)
        connect(out_group["software.learn_new_object.done"], self.input_from_yarp["state_machine.learn_new_object.done"], synaptic_weight=cos_weight)
        connect(out_group["software.query_memory.done"], self.input_from_yarp["state_machine.query_memory.done"], synaptic_weight=cos_weight)

def create_simulated_input(net, timesteps):
    
    TIMESTEPS = timesteps
    NEURONS_PER_NODE = 1
    
    inputs = []
    out_groups = {}
    
    cos_weight = 0.6 * Node.THRESHOLD
    
    cos_look_at_object_done_name = "software.look_at_object.done"
    cos_look_at_object_done = Input(cos_look_at_object_done_name, net, NEURONS_PER_NODE, TIMESTEPS)
    cos_look_at_object_done.create_complete_input(TIMESTEPS, 10, 1)
    inputs.append(cos_look_at_object_done)
    out_groups[cos_look_at_object_done_name] = cos_look_at_object_done
    
    cos_recognize_object_known_name = "object_recognition.recognize_object.known"
    cos_recognize_object_known = Input(cos_recognize_object_known_name, net, NEURONS_PER_NODE, TIMESTEPS)
    #cos_recognize_object_known.create_complete_input(TIMESTEPS, 17, 1)
    cos_recognize_object_known.create_empty_input(TIMESTEPS)
    inputs.append(cos_recognize_object_known)
    out_groups[cos_recognize_object_known_name] = cos_recognize_object_known
    
    cos_recognize_object_unknown_name = "object_recognition.recognize_object.unknown"
    cos_recognize_object_unknown = Input(cos_recognize_object_unknown_name, net, NEURONS_PER_NODE, TIMESTEPS)
    cos_recognize_object_unknown.create_complete_input(TIMESTEPS, 17, 1)
    #cos_recognize_object_unknown.create_empty_input(TIMESTEPS)
    inputs.append(cos_recognize_object_unknown)
    out_groups[cos_recognize_object_unknown_name] = cos_recognize_object_unknown
    
    cos_learn_new_object_done_name = "software.learn_new_object.done"
    cos_learn_new_object_done = Input(cos_learn_new_object_done_name, net, NEURONS_PER_NODE, TIMESTEPS)
    cos_learn_new_object_done.create_complete_input(TIMESTEPS, 25, 1)
    #cos_learn_new_object_done.create_empty_input(TIMESTEPS)
    inputs.append(cos_learn_new_object_done)
    out_groups[cos_learn_new_object_done_name] = cos_learn_new_object_done
    
    cos_query_memory_done_name = "software.query_memory.done"
    cos_query_memory_done = Input(cos_query_memory_done_name, net, NEURONS_PER_NODE, TIMESTEPS)
    cos_query_memory_done.create_complete_input(TIMESTEPS, 15, 1)
    inputs.append(cos_query_memory_done)
    out_groups[cos_query_memory_done_name] = cos_query_memory_done
    
    return out_groups, inputs



def create_visualization(all_behaviors, behavior_dictionary, timesteps):

    def draw_axon(neuron1, neuron2, inhibitory=False, curved=False, pale=False):
        radius = neuron1.radius
        angle = math.atan2(neuron2.x - neuron1.x, neuron2.y - neuron1.y)
        x_adjustment = radius * math.sin(angle)
        y_adjustment = radius * math.cos(angle)
        
        if (pale):
            color = "pink" if (inhibitory) else "xkcd:very light green"
        else:
            color = "tab:red" if (inhibitory) else "tab:green"
        
        connection_style = "arc3,rad=.5" if (curved) else None
        
        kw = dict(arrowstyle="Simple, tail_width=1, head_width=4, head_length=4")
        line = matplotlib.patches.FancyArrowPatch((neuron1.x + x_adjustment, neuron1.y + y_adjustment),
                                                  (neuron2.x - x_adjustment, neuron2.y - y_adjustment),
                                                  connectionstyle=connection_style,
                                                  zorder=1,
                                                  fc=color,
                                                  ec=color,
                                                  **kw)
        
        
        matplotlib.pyplot.gca().add_line(line)
    
    def draw_behavior(x, y, behavior):
        offset_x_neurons = 2
        offset_y_neurons = 2

        # behavior label
        offset_x_center = (len(behavior.nodes_cos) * offset_x_neurons) / 2.
        matplotlib.pyplot.text(x + offset_x_center,
                               y - 2 * offset_y_neurons,
                               behavior.name.replace("state_machine.", "").replace("_", " "),
                               ha='center',
                               va='bottom',
                               transform=ax.transData)


        # intention label
        #matplotlib.pyplot.text(x,
        #                       y + offset_y_neurons,
        #                       "start",
        #                       ha='center',
        #                       va='bottom',
        #                       transform=ax.transData)
        
        # prior intention
        behavior.node_prior_intention.visualization = Node.Visualization(x, y, False)
        behavior.node_prior_intention.visualization.draw()
        
         # intention
        behavior.node_intention.visualization = Node.Visualization(x, y-offset_y_neurons, False)
        behavior.node_intention.visualization.draw()
        
        draw_axon(behavior.node_prior_intention.visualization, behavior.node_intention.visualization)
        
        for j,key in enumerate(behavior.nodes_cos.keys()):
            x += offset_x_neurons
            
            # cos label
            #matplotlib.pyplot.text(x,
            #                       y + offset_y_neurons,
            #                       key,
            #                       ha='center',
            #                       va='bottom',
            #                       transform=ax.transData)
            
             # cos memory
            node_cos_memory = behavior.nodes_cos_memory[key]
            node_cos_memory.visualization = Node.Visualization(x, y, False)
            node_cos_memory.visualization.draw()
            # cos
            node_cos = behavior.nodes_cos[key]
            node_cos.visualization = Node.Visualization(x, y-offset_y_neurons, False)
            node_cos.visualization.draw()
            
            curved = True if (j > 0) else False
            draw_axon(behavior.node_intention.visualization, node_cos.visualization, curved=curved)
            draw_axon(node_cos_memory.visualization, behavior.node_intention.visualization, inhibitory=True)
            draw_axon(node_cos.visualization, node_cos_memory.visualization)
        
        return x
    
    def update_behavior_state(behavior, timestep):
        active_pi = behavior.node_prior_intention.probe_spikes.data[0,timestep] > 0.5
        behavior.node_prior_intention.visualization.update(active_pi)
        
        active_i = behavior.node_intention.probe_spikes.data[0,timestep] > 0.5
        behavior.node_intention.visualization.update(active_i)
        
        for key in behavior.nodes_cos.keys():
            node_cos_memory = behavior.nodes_cos_memory[key]
            active_cm = node_cos_memory.probe_spikes.data[0,timestep] > 0.5
            node_cos_memory.visualization.update(active_cm)

            node_cos = behavior.nodes_cos[key]
            active_c = node_cos.probe_spikes.data[0,timestep] > 0.5
            node_cos.visualization.update(active_c)
        
    def draw_precondition_axons(precondition):
        precondition_neuron = precondition.visualization
        
        for behavior_name,cos_name in precondition_neuron.sources.items(): 
            source = behavior_dictionary[behavior_name].nodes_cos_memory[cos_name].visualization
            draw_axon(source, precondition_neuron, inhibitory=True)
            
        target = behavior_dictionary[precondition_neuron.target].node_intention.visualization
        draw_axon(precondition_neuron, target, inhibitory=True)
        
        draw_axon(precondition_neuron.task_node.visualization, precondition_neuron, pale=True)
        
    def draw_subbehavior_axons(behavior):
        
        for name in behavior.visualization.subbehaviors:
            subbehavior = behavior_dictionary[name]
            
            draw_axon(behavior.node_intention.visualization, subbehavior.node_prior_intention.visualization, pale=True)
            draw_axon(subbehavior.node_prior_intention.visualization, behavior.nodes_cos["done"].visualization, inhibitory=True, pale=True)
            
            for cos_name in subbehavior.nodes_cos_memory.keys():
                draw_axon(subbehavior.nodes_cos_memory[cos_name].visualization, behavior.nodes_cos["done"].visualization, pale=True)
    
    def draw_architecture(layers):
        
        offset_x_behaviors = 3
        offset_y_behaviors = 10
        offset_y_preconditions = 2
        
        x = x_init = 1
        y = 3
        
        for l,layer in enumerate(layers):

            # aligning the layers horizontally
            if (l == 1):
                x += 8

            for behavior in layer:
                if (behavior.name == "state_machine.query_memory"):
                    x += 3
                elif (behavior.name == "query"):
                    x += 13
                
                for j,precondition in enumerate(behavior.preconditions.values()):
                    precondition.visualization.x = x
                    precondition.visualization.y = y+((j+1) * offset_y_preconditions)
                    precondition.visualization.active = False
                    precondition.visualization.draw()

                x += offset_x_behaviors
                x = draw_behavior(x, y, behavior)
                x += offset_x_behaviors

            x = x_init
            y += offset_y_behaviors

        for layer in layers:
            for behavior in layer:
                if (behavior.has_subbehaviors):
                    draw_subbehavior_axons(behavior)

                for precondition in behavior.preconditions.values():
                    draw_precondition_axons(precondition)

    def update_architecture_state(timestep):
        
        plt.axis('scaled')
        plt.axis('off')
        plt.axis([0, 60, 0, 25])
        
        for l,layer in enumerate(layers):
            for behavior in layer:
                for precondition in behavior.preconditions.values():
                    active = precondition.probe_spikes.data[0,timestep] > 0.5
                    
                    precondition.visualization.update(active)

                update_behavior_state(behavior, timestep)
    
    
    fig = plt.figure(figsize=(14,7))
    #fig.set_dpi(15)
    ax = fig.gca()
    plt.cla()
    plt.axis('scaled')
    plt.axis('off')
    plt.axis([2, 45, 0, 15])
    
    # layer 2
    beh_look = behavior_dictionary["look"]
    beh_query = behavior_dictionary["query"]
    layer2 = [beh_look, beh_query]
    
    beh_look.visualization.subbehaviors.remove("state_machine.dummy")

    # layer 1
    beh_look_obj = behavior_dictionary["state_machine.look_at_object"]
    beh_rec_obj = behavior_dictionary["state_machine.recognize_object"]
    beh_learn_obj = behavior_dictionary["state_machine.learn_new_object"]
    beh_query_mem = behavior_dictionary["state_machine.query_memory"]
    layer1 = [beh_look_obj, beh_rec_obj, beh_learn_obj, beh_query_mem]

    layers = [layer1, layer2]
    
    draw_architecture(layers)
    
    
    #fig.canvas.draw()
    #fig.canvas.flush_events()
    plt.show()
    

    animation = matplotlib.animation.FuncAnimation(fig, update_architecture_state, frames=30, repeat=True)

    #plt.show()
    
    #HTML(animation.to_html5_video())
    #display(HTML(animation.to_html5_video()))
    
    animation.save("animation.gif", writer='imagemagick', fps=3)
