import nxsdk.api.n2a as nx

from dft_loihi.visualization.plotting import Plotter
from dft_loihi.dft.behavior import Behavior

# for visualization
import matplotlib.animation
from matplotlib import rc
from IPython.display import HTML, Image, display
rc('animation', html='html5')
#matplotlib.pyplot.rcParams['animation.convert_path'] = '/homes/mathis.richter/programs/imagemagick_install/bin/magick'


class StateMachine():
        
    def __init__(self, net):
        
        self.net = net

        # return values
        self.probed_groups = {}   # a dictionary of all compartments that produce output for YARP
        self.out_groups = {}   # a dictionary of all compartments that produce output to other networks on Loihi

        self.input_from_yarp = {}  # a dictionary of all compartments that receive input from YARP
        self.input_from_loihi = {}  # a dictionary of all compartments that receive input from other networks on Loihi

        self.behavior_dictionary = {}
        
        self.groups = {} # a dictionary of all the compartment groups

        behavior_look = self.create_behavior(
            "look",
            output=self.probed_groups,
            has_subbehaviors=True)

        behavior_look_at_object = self.create_behavior(
            "state_machine.look_at_object",
            cos_names=["done"],
            output=self.probed_groups,
            input=self.input_from_yarp)

        behavior_recognize_object = self.create_behavior(
            "state_machine.recognize_object",
            cos_names=["known", "unknown"],
            output=self.probed_groups,
            input=self.input_from_loihi)

        behavior_learn_new_object = self.create_behavior(
            "state_machine.learn_new_object",
            cos_names=["done"],
            output=self.probed_groups,
            input=self.input_from_yarp)
        
        behavior_dummy = self.create_behavior(
            "state_machine.dummy",
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
            output=self.probed_groups,
            has_subbehaviors=True)

        behavior_query_memory = self.create_behavior(
            "state_machine.query_memory",
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
        node_look = Node(node_look_name, net, biasMant=look_bias, biasExp=7)
        self.groups[node_look_name] = node_look
        behavior_look.add_boost(node_look)
        connect(behavior_look.nodes_cos_memory["done"], node_look, synaptic_weight=-1.5 * Node.THRESHOLD)

        node_query_name = "Node query"
        node_query = Node(node_query_name, net, biasMant=query_bias, biasExp=7)
        self.groups[node_query_name] = node_query
        behavior_query.add_boost(node_query)
        connect(behavior_query.nodes_cos_memory["done"], node_query, synaptic_weight=-1.5 * Node.THRESHOLD)
        
        # for visualization
        #self.probes = {}
        #for behavior in self.behaviors:
        #    if (not behavior.has_subbehaviors):
        #        register_probes(behavior, self.probes)
    
    def create_behavior(self, name, cos_names=None, has_subbehaviors=False, output=None, input=None):
        behavior = dft_loihi.dft.behavior.Behavior(name, self.net, has_subbehaviors=has_subbehaviors)
        
        if (cos_names != None):
            for cos_name in cos_names:
                behavior.add_cos(cos_name)
        
        register_behavior(behavior, self.groups, self.behavior_dictionary, output=output, input=input)
        
        return behavior

    def register_behavior(self, behavior, groups, behavior_dictionary, output=None, input=None):
        behavior_dictionary[behavior.name] = behavior
        
        groups[behavior.name + ".prior_intention"] = behavior.node_prior_intention.neurons
        groups[behavior.name + ".intention"] = behavior.node_intention.neurons
        
        if (output != None):
            output[behavior.name + ".start"] = behavior.node_intention.neurons
        
        for key in behavior.nodes_cos.keys():
            groups[behavior.name + ".cos." + key] = behavior.nodes_cos[key].neurons
            groups[behavior.name + ".cos_memory." + key] = behavior.nodes_cos_memory[key].neurons
            
            if (input != None):
                input[behavior.name + "." + key] = behavior.nodes_cos[key].neurons
                
    #def register_probes(self, behavior, probes):
    #    probes[behavior.name + ".prior_intention"] = behavior.node_prior_intention.probe_spikes
    #    probes[behavior.name + ".intention"] = behavior.node_intention.probe_spikes
    #    
    #    for name,node in behavior.nodes_cos.items():
    #        probes[behavior.name + ".cos." + name] = node.probe_spikes
    #    
    #    for name,node in behavior.nodes_cos_memory.items():
    #        probes[behavior.name + ".cos_memory." + name] = node.probe_spikes
    #        
    #    for name,node in behavior.preconditions.items():
    #        probes[behavior.name + ".precondition." + name] = node.probe_spikes
    
    def connect_in(self, out_group):
        cos_weight = 0.6 * Node.THRESHOLD

        connect(out_group["software.look_at_object.done"], self.input_from_yarp["state_machine.look_at_object.done"], synaptic_weight=cos_weight)
        connect(out_group["object_recognition.recognize_object.known"], self.input_from_loihi["state_machine.recognize_object.known"], synaptic_weight=cos_weight)
        connect(out_group["object_recognition.recognize_object.unknown"], self.input_from_loihi["state_machine.recognize_object.unknown"], synaptic_weight=cos_weight)
        connect(out_group["software.learn_new_object.done"], self.input_from_yarp["state_machine.learn_new_object.done"], synaptic_weight=cos_weight)
        connect(out_group["software.query_memory.done"], self.input_from_yarp["state_machine.query_memory.done"], synaptic_weight=cos_weight)

def create_simulated_input(net, timesteps):
    
    neurons_per_node = 1
    
    out_groups = {}
    cos_weight = 0.6 * Node.THRESHOLD
    
    cos_look_at_object_done_name = "software.look_at_object.done"
    cos_look_at_object_done = Input(cos_look_at_object_done_name, net, neurons_per_node, timesteps)
    cos_look_at_object_done.create_complete_input(timesteps, 10, 1)
    out_groups[cos_look_at_object_done_name] = cos_look_at_object_done
    
    cos_recognize_object_known_name = "object_recognition.recognize_object.known"
    cos_recognize_object_known = Input(cos_recognize_object_known_name, net, neurons_per_node, timesteps)
    #cos_recognize_object_known.create_complete_input(timesteps, 17, 1)
    cos_recognize_object_known.create_empty_input(timesteps)
    out_groups[cos_recognize_object_known_name] = cos_recognize_object_known
    
    cos_recognize_object_unknown_name = "object_recognition.recognize_object.unknown"
    cos_recognize_object_unknown = Input(cos_recognize_object_unknown_name, net, neurons_per_node, timesteps)
    cos_recognize_object_unknown.create_complete_input(timesteps, 17, 1)
    #cos_recognize_object_unknown.create_empty_input(timesteps)
    out_groups[cos_recognize_object_unknown_name] = cos_recognize_object_unknown
    
    cos_learn_new_object_done_name = "software.learn_new_object.done"
    cos_learn_new_object_done = Input(cos_learn_new_object_done_name, net, neurons_per_node, timesteps)
    cos_learn_new_object_done.create_complete_input(timesteps, 25, 1)
    #cos_learn_new_object_done.create_empty_input(timesteps)
    out_groups[cos_learn_new_object_done_name] = cos_learn_new_object_done
    
    cos_query_memory_done_name = "software.query_memory.done"
    cos_query_memory_done = Input(cos_query_memory_done_name, net, neurons_per_node, timesteps)
    cos_query_memory_done.create_complete_input(timesteps, 15, 1)
    out_groups[cos_query_memory_done_name] = cos_query_memory_done
    
    return out_groups



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



def main():
    TIMESTEPS = 30

    net = nx.NxNet()

    state_machine = StateMachine(net)

    out_groups = create_simulated_input(net, TIMESTEPS)
    state_machine.connect_in(out_groups)

    net.run(TIMESTEPS)
    net.disconnect()

    #create_visualization(state_machine.behaviors, state_machine.behavior_dictionary, TIMESTEPS)

    behavior_names = ["look",
            "state_machine.look_at_object",
            "state_machine.recognize_object",
            "state_machine.learn_new_object",
            "state_machine.dummy",
            "state_machine.recognize_object",
            "query",
            "state_machine.query_memory"]

    #plotter = dft.visualization.Plotter()
    #for name in behavior_names:
    #    plotter.plot_behavior(state_machine.behavior_dictionary[name])
