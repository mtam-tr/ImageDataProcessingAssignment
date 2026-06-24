import numpy as np
from neuralNetwork import Neural_network

class Agent:
    def __init__(self):
        self.net = Neural_network([2, 1])

        self.activations = [
            'arctan',
            'id',
            'id_courb',
            'sigmoid',
            'sin',
            'sinc',
            'softmax',
            'softplus',
            'softsign',
            'swish',
            'tanh'
        ]
        self.current_preset = 3

        self.activation_functions = [None, self.activations[self.current_preset]]

        # Fixed bias
        self.b = 0
        self.net.set_biases([self.b])

    def next_preset(self):
        self.current_preset = (self.current_preset + 1) % len(self.activations)
        self.activation_functions[1] = self.activations[self.current_preset]

    def update(self):
        pass

    def draw(self, interface, screen, size, cell):
        max_pos = size - cell

        for py in range(0, size, cell):
            # y in [-1, +1]
            y = 2.0 * (py / max_pos - 0.5)

            for px in range(0, size, cell):
                # x in [-1, +1]
                x = 2.0 * (px / max_pos - 0.5)

                out = self.net.update([x, y], self.activation_functions)[0]

                # handle tuple return from neural network
                if isinstance(out, tuple):
                    out = out[0]
                
                g = int(np.clip(255 * out, 0, 255))
                r = 255 - g
                b = 255 - g
                color = (r, g, b)

                interface.draw.rect(screen, color, (px, py, cell, cell))