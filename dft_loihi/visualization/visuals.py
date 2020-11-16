
    # for a node or neuron
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

    # for a behavior
    class Visualization():
        def __init__(self):
            self.subbehaviors = []
  
