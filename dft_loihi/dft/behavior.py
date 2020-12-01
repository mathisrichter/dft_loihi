# Mathis Richter (mathis.richter@ini.rub.de), 2020

#import math
#import matplotlib.pyplot as plt
#import numpy as np
#import nxsdk.api.n2a as nx
#import nxsdk.net.groups
#from nxsdk.utils.plotutils import plotRaster
#from time import sleep

from dft_loihi.dft.node import Node
from dft_loihi.dft.util import connect


class Behavior:
    
    """A small network of dynamic neural nodes that represent and control a cognitive operation (or behavior)."""
    def __init__(self, name, net, has_subbehaviors=False):
        """Constructor
        
        Parameters:
        name --- A string that describes the behavior.
        net --- The nxsdk network object required to create the neurons.
        has_subbehaviors --- Set this flag to true if this behavior does not do any direct work but only connects to a set of subbehaviors.
        """
        self.name = name
        self.net = net
        self.has_subbehaviors = has_subbehaviors
        
        # activated and sustained by input from task (or higher-level intention node), turns off without that input
        self.node_prior_intention = Node("Prior Intention", net)
        # activated by input from prior intention node, turns off without that input
        self.node_intention = Node("Intention", net)
        
        # a dictionary for CoS memory nodes
        self.nodes_cos_memory = {}
        # a dictionary for CoS nodes
        self.nodes_cos = {}
        # a dictionary for precondition nodes
        self.preconditions = {}

        # the prior intention node activates the intention node
        connect(self.node_prior_intention, self.node_intention, 1.1)
        
        # if this behavior only has subbehaviors, add a single CoS signal
        # it will later be fed from the CoS signals of all subbehaviors
        if (self.has_subbehaviors):
            single_cos_name = "done"
            self.add_cos(single_cos_name)
            self.single_node_cos = self.nodes_cos[single_cos_name]

        #self.visualization = Behavior.Visualization()
    
    def add_cos(self, cos_name, cos_input=None):
        """Adds a condition-of-satisfaction (CoS) to the behavior.
        
        Parameters:
        name --- A string that describes the CoS.
        cos_input --- The input that provides the CoS signal. Leave empty if you want to add it manually later.
        """
        # activated by input from BOTH intention node and "sensory" input, turns off without either
        cos_node = Node(cos_name, self.net)
        self.nodes_cos[cos_name] = cos_node
        # activated by input from BOTH task and cos node, remains on without cos input, turns off without task input
        cos_memory_node = Node(cos_name + " Memory", self.net, self_excitation=0.3)
        self.nodes_cos_memory[cos_name] = cos_memory_node
        
        int_cos_weight = 0.6 if (not self.has_subbehaviors) else 1.0
        connect(self.node_intention, cos_node, int_cos_weight)
        connect(cos_node, cos_memory_node, 0.3)
        connect(cos_memory_node, self.node_intention, -1.0)
        
        if (cos_input != None):
            connect(cos_input, cos_node, 0.6)
        
    def add_boost(self, boost_input):
        """Adds an artificial boost input to the behavior, which may activate it; relevant for top-level behaviors."""
        connect(boost_input, self.node_prior_intention, 1.1)
        for cos_memory in self.nodes_cos_memory.values():
            connect(boost_input, cos_memory, 0.8)
        
    def add_subbehaviors(self, sub_behaviors, logical_or=False):
        """Connects a list of other behaviors in such a way that they are on the next lower hierarchical level
        with respect to the current behavior.
        
        Parameters:
        sub_behaviors --- A list of subbehaviors. By default, all their CoS signals have to be achieved
                          in order to trigger this behavior's CoS.
        logical_or --- By setting this flag to true, this behavior requires only one of the subbehaviors' CoS to finish.
        """
        for sub_behavior in sub_behaviors:
            connect(self.node_intention, sub_behavior.node_prior_intention, 1.1)
            
            weight = -1.1
            if logical_or:
                weight = weight / len(sub_behaviors)
            connect(sub_behavior.node_prior_intention, self.single_node_cos, weight)
            
            for sub_cos_memory in sub_behavior.nodes_cos_memory.values():
                connect(self.node_intention, sub_cos_memory, 0.8)
                connect(sub_cos_memory, self.single_node_cos, 1.2)
                
            #self.visualization.subbehaviors.append(sub_behavior.name)
        
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
        precondition_node = Node(name, self.net)
        self.preconditions[name] = precondition_node
        if (register_groups != None):
            register_groups[name] = precondition_node.neurons
        connect(precondition_node, self.node_intention, -1.1)
        #precondition_node.visualization = Node.PreconditionVisualization()
        #precondition_node.visualization.set_target_behavior(self.name)
        #precondition_node.visualization.task_node = task_input
        for i in range(len(precondition_behaviors)):
            for j in range(len(cos_names[i])):
                weight = -1.1
                if (not logical_or):
                    weight = weight / len(precondition_behaviors)
                connect(precondition_behaviors[i].nodes_cos_memory[cos_names[i][j]],
                        precondition_node,
                        weight)
                #precondition_node.visualization.add_source_cos(precondition_behaviors[i].name, cos_names[i][j])
        connect(task_input, precondition_node, 1.1)
