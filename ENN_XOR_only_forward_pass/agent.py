from neuralNetwork import Neural_network

class Agent():    
    def __init__(self, inputs):
        self.neural_net = Neural_network([2, 2, 1])
        self.inputs = inputs
 
    # Forward pass
    def update(self):
        self.outputs = self.neural_net.update(self.inputs)        
            
    def draw(self, interface, screen):       
        # Initialize the graph (E, V) of the neural network
        delta_x = 0.2 * screen.get_width()
        delta_y = 0.25 * screen.get_height()
        offset_left = 0.15 * screen.get_width()
        radius = 20
        y_center = screen.get_height() // 2
        
        # Initialize line and node data
        lines = []
        nodes = []
        
        # Store lines
        for layer_pair_it, (left_size, right_size) in enumerate(zip(self.neural_net.layer_sizes[:-1], self.neural_net.layer_sizes[1:])):
            y_bottom_left = y_center + 0.5 * delta_y * (left_size - 1)
            y_bottom_right = y_center + 0.5 * delta_y * (right_size - 1)
            
            for left_node in range(left_size):
                for right_node in range(right_size):      
                    line = [
                        [layer_pair_it * delta_x + offset_left, 
                         y_bottom_left - left_node * delta_y],
                        [(layer_pair_it + 1) * delta_x + offset_left, 
                         y_bottom_right - right_node * delta_y]
                    ]
                    lines.append((line, (0, 0, 0), 2))
    
        # Draw lines
        screen.fill(interface.Color('white')) 
        for line, color, width in lines:
            interface.draw.lines(screen, color, False, line, width)
        
        # Store nodes        
        for layer_number, layer_size in enumerate(self.neural_net.layer_sizes):
            y_bottom = y_center + 0.5 * delta_y * (layer_size - 1)
            for node in range(layer_size):
                color = (0, (layer_number == 0) * 255, (layer_number == 1) * 255)
                x, y = (layer_number * delta_x + offset_left, y_bottom - node * delta_y)
                nodes.append(((x, y), color, radius))
        
        # Draw nodes
        for pos, color, rad in nodes:
            interface.draw.circle(screen, color, pos, rad)
            interface.draw.circle(screen, interface.Color('black'), pos, rad, width = 2)
            
        # Draw text
        font_renderer = interface.font.SysFont('Consolas', int(0.05 * screen.get_width()))
        text = f'input = {self.inputs}, output = {self.outputs[0]:.4f}'
        string = font_renderer.render(text, True, interface.Color('black'))
        screen.blit(string, (radius, radius + string.get_height()))