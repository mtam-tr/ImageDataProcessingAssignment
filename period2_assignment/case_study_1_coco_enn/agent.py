import numpy as np
from neuralNetwork import Neural_network
import sys
sys.dont_write_bytecode = True

class Agent():
    caption = 'Case 1 - CoCo ENN'
    window_size = (600, 620)
    frames_per_second = 30
    population_size = 40

    
    training_pixels = [
        ((2, 2), np.array([0.0, 0.0, 1.0])),    # blue
        ((2, 13), np.array([0.0, 1.0, 0.0])),   # green
        ((13, 13), np.array([1.0, 0.0, 0.0])),  # red
    ]

    def __init__(self):
        # x/y coordinates in, RGB color out
        self.neural_net = Neural_network([2, 10, 3])
        self.fitness = 0
        self.error = float('inf')

    def predict_color(self, x, y):
        inputs = [x / 15.0, y / 15.0] # normalize to the 16x16 grid
        color = self.neural_net.update(inputs)
        return np.array(color) 

    def update(self):
        # measure performance
        total_error = 0.0
        for (x, y), target_color in Agent.training_pixels:
            predicted_color = self.predict_color(x, y)
            total_error += np.mean((predicted_color - target_color) ** 2)

        self.error = total_error / len(Agent.training_pixels)
        # convert errors into a fitness score
        self.fitness = 1.0 / (1.0 + 10.0 * self.error)

    def draw(self, interface, screen, generation):
        font = interface.font.SysFont('Consolas', 22)
        small_font = interface.font.SysFont('Consolas', 16)

        screen.blit(font.render('Coordinates-to-Color ENN', True, (20, 20, 20)), (32, 24))
        screen.blit(small_font.render(f'generation: {generation}', True, (20, 20, 20)), (36, 56))
        screen.blit(small_font.render(f'training MSE: {self.error:.6f}', True, (20, 20, 20)), (220, 56))

        left = 60
        top = 96
        cell = 30

        for y in range(16):
            for x in range(16):
                color = tuple((255 * self.predict_color(x, y)).astype(int))
                rect = interface.Rect(left + x * cell, top + y * cell, cell, cell)
                interface.draw.rect(screen, color, rect)

        # mark the known training pixels
        for (x, y), target_color in Agent.training_pixels:
            rect = interface.Rect(left + x * cell, top + y * cell, cell, cell)
            interface.draw.rect(screen, tuple((255 * target_color).astype(int)), rect.inflate(-10, -10))
