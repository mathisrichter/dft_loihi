    # register groups of a behavior
    def register_behavior(self, groups, behavior_dictionary, output=None, input=None):
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

    # register probes of a behavior
    def register_probes(self, probes):
        probes[self.name + ".prior_intention"] = self.node_prior_intention.probe_spikes
        probes[self.name + ".intention"] = self.node_intention.probe_spikes
        
        for name,node in self.nodes_cos.items():
            probes[self.name + ".cos." + name] = node.probe_spikes
        
        for name,node in self.nodes_cos_memory.items():
            probes[self.name + ".cos_memory." + name] = node.probe_spikes
            
        for name,node in self.preconditions.items():
            probes[self.name + ".precondition." + name] = node.probe_spikes
  
