import numpy as np
import dft_loihi.dft.util

class SimulatedInput(dft_loihi.dft.util.Connectable):
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
        
        self.simulated_input = net.createSpikeGenProcess(numPorts=self.number_of_neurons)

        # for connections
        self.output = self.simulated_input
        
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
        
       
    def create_input(self):
        """Creates the overall time course of the input.
        Make sure to only call this once you have added all phases of input with the add_input_phase() function."""
        
        concatenated_phases = np.empty((self.number_of_neurons, 0))
        
        for phase in self.input_phases:
            concatenated_phases = np.append(concatenated_phases, phase, axis=1)
                
        for i in range(self.number_of_neurons):
            st = np.where(concatenated_phases[i,:])[0].tolist()
            self.spike_times.append(st)
            self.simulated_input.addSpikes(spikeInputPortNodeIds=i, spikeTimes=st)
       
    def create_empty_input(self, length):
        self.input_phases.append(np.zeros((self.number_of_neurons, length)))
        self.create_input()
        
    def create_complete_input(self, length, start_on, length_on):
        self.add_input_phase_on_off(start_on-1, on=False)
        self.add_input_phase_on_off(length_on, on=True)
        self.add_input_phase_on_off(length-(start_on-1)-length_on, on=False)
        self.create_input()
