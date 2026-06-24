import numpy as np
from neuralNetwork import Neural_network


class Agent():
    caption = 'Case 1 - CoCo ENN'
    window_size = (600, 620)
    frames_per_second = 30
    population_size = 40

    # The assignment gives only three known pixels. The ENN learns from these
    # examples and predicts the colors of the remaining 16x16 pixels.
    training_pixels = [
        ((2, 2), np.array([0.0, 0.0, 1.0])),    # blue
        ((2, 13), np.array([0.0, 1.0, 0.0])),   # green
        ((13, 13), np.array([1.0, 0.0, 0.0])),  # red
    ]

    def __init__(self):
        # Simple network: x/y coordinate in, RGB color out.
        self.neural_net = Neural_network([2, 3])
        self.activation_functions = [None, 'sigmoid']
        self.fitness = 0
        self.error = float('inf')

    def predict_color(self, x, y):
        inputs = [x / 15.0, y / 15.0]
        color = self.neural_net.update(inputs, self.activation_functions)
        return np.clip(np.array(color), 0.0, 1.0)

    def update(self):
        # Fitness is higher when the three known pixels are predicted correctly.
        total_error = 0.0
        for (x, y), target_color in Agent.training_pixels:
            predicted_color = self.predict_color(x, y)
            total_error += np.mean((predicted_color - target_color) ** 2)

        self.error = total_error / len(Agent.training_pixels)
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
                interface.draw.rect(screen, (230, 230, 230), rect, 1)

        # Mark the three original training pixels with a black border.
        for (x, y), target_color in Agent.training_pixels:
            rect = interface.Rect(left + x * cell, top + y * cell, cell, cell)
            interface.draw.rect(screen, (0, 0, 0), rect, 3)
            interface.draw.rect(screen, tuple((255 * target_color).astype(int)), rect.inflate(-10, -10))

        screen.blit(small_font.render('black border = known training pixel', True, (20, 20, 20)), (60, 590))
