import sys
import numpy as np
from collections.abc import Iterable
import dft_loihi.dft.util
from dft_loihi.dft.util import time_steps_per_minute
from dft_loihi.dft.util import gauss


class PiecewiseStaticInput(dft_loihi.dft.util.Connectable):
    def __init__(self, name, net, shape):
        self.name = name

        if type(shape) == int:
            shape = (shape,)

        self.shape = shape
        self.number_of_neurons = int(np.prod(self.shape))
        self.piecewise_static_input = net.createSpikeGenProcess(numPorts=self.number_of_neurons)

        self.spikes = np.empty((self.number_of_neurons, 0))
        self.spike_times = []  # for plotting
        self.number_of_time_steps = 0

        # for connections
        self.output = self.piecewise_static_input

    def create(self):
        """Creates the overall time course of the input.
        Make sure to only call this once you have added all phases of input with the add_input_phase() function."""

        for i in range(self.number_of_neurons):
            spike_times = np.nonzero(self.spikes[i, :])[0].tolist()
            self.spike_times.append(spike_times)
            self.piecewise_static_input.addSpikes(spikeInputPortNodeIds=i,
                                                  spikeTimes=spike_times)

        self.number_of_time_steps = np.size(self.spikes, axis=1)


class GaussPiecewiseStaticInput(PiecewiseStaticInput):
    def __init__(self, name, net, shape, domain=None):
        super().__init__(name, net, shape)

        if type(shape) == int:
            shape = (shape,)
            if domain is not None:
                domain = np.array([domain], dtype=np.float32)

        if domain is None:
            domain = np.zeros((len(shape), 2), dtype=np.float32)
            domain[:, 1] = shape[:]

        self.domain = domain

    def add_spike_rate(self, max_spike_rate, center, width, duration):
        """Spike rate is in Hz (spikes per minute)
        center and width refer to the domain of the field
        duration is in time steps"""
        if not isinstance(center, Iterable):
            center = [center]
            width = [width]

        assert len(center) == len(self.shape)
        assert len(width) == len(self.shape)

        spike_rate_per_time_step = max_spike_rate / time_steps_per_minute
        spike_rates = gauss(self.domain,
                            self.shape,
                            spike_rate_per_time_step,
                            center,
                            width)

        spike_rates = spike_rates.flatten()
        spike_rates = np.asarray([spike_rates, ])

        spikes = np.random.poisson(spike_rates.T, (self.number_of_neurons, duration))
        spikes = np.where(spikes > 0, 1, spikes)

        self.spikes = np.append(self.spikes, spikes, axis=1)


class HomogeneousPiecewiseStaticInput(PiecewiseStaticInput):
    def __init__(self, name, net, shape):
        super().__init__(name, net, shape)

    def add_spike_rate(self, spike_rate, duration):
        """Spike rate is in Hz (spikes per minute)."""
        spike_rate_per_time_step = spike_rate / time_steps_per_minute

        spikes = np.random.poisson(spike_rate_per_time_step, (self.number_of_neurons, duration))
        spikes = np.where(spikes > 0, 1, spikes)

        self.spikes = np.append(self.spikes, spikes, axis=1)



# TODO merge this into PiecewiseStaticInput
class SimulatedInput(dft_loihi.dft.util.Connectable):
    """
    An artificial spike pattern that can be used as an input to nodes.
    The pattern can be shaped by specifying phases of time in which all neurons have a certain probability of spiking at every time step.
    
    After creating the input object, add a number of input phases (add_input_phase function) and then create the overall input time couse (create_input() function.
    """
    
    def __init__(self, name, net, number_of_neurons, number_of_time_steps):
        """Constructor
        
        Parameters:
        name --- A string that describes the input.
        net --- The nxsdk network object required to create the neurons.
        number_of_neurons --- Number of neurons for this input.
        number_of_timesteps --- The total number of time steps over which the input time course will be generated."""
        self.name = name
        self.number_of_neurons = number_of_neurons
        self.number_of_time_steps = number_of_time_steps
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

    def add_input_phase_spike_rate(self, length, spike_rate):
        """Spike rate is in Hz (spikes per minute)."""
        time_steps_per_minute = 600

        if (spike_rate == 0.0):
            spike_distance = sys.maxsize
        else:
            spike_distance = round(time_steps_per_minute / spike_rate)
        self.add_input_phase_spike_distance(length, spike_distance)
    
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


