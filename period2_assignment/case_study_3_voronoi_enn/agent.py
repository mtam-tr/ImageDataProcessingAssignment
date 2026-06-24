import numpy as np
from neuralNetwork import Neural_network


class Agent():
    caption = 'Case 3 - Voronoi ENN'
    window_size = (760, 650)
    frames_per_second = 18
    population_size = 40
    centroids = np.array([
        [0.18, 0.82],
        [0.22, 0.28],
        [0.50, 0.58],
        [0.78, 0.80],
        [0.82, 0.22],
    ])
    colors = [
        (255, 0, 128),
        (0, 220, 235),
        (0, 235, 40),
        (245, 235, 0),
        (135, 0, 235),
    ]
    training_points = np.array([[x, y] for y in np.linspace(0, 1, 13) for x in np.linspace(0, 1, 13)])

    def __init__(self):
        self.neural_net = Neural_network([2, 14, 5])
        self.activation_functions = [None, 'tanh', 'id']
        self.fitness = 0
        self.error = float('inf')
        self.accuracy = 0

    @staticmethod
    def nearest_centroid(point):
        distances = np.sum((Agent.centroids - point) ** 2, axis=1)
        return int(np.argmin(distances))

    def predict_label(self, point):
        outputs = self.neural_net.update(point, self.activation_functions)
        return int(np.argmax(outputs)), np.asarray(outputs, dtype=float)

    def update(self):
        # The target labels are generated analytically from the Voronoi rule.
        # Fitness rewards both classification accuracy and one-hot output shape.
        total_error = 0.0
        correct = 0
        for point in Agent.training_points:
            target_label = Agent.nearest_centroid(point)
            predicted_label, outputs = self.predict_label(point)
            target = np.zeros(5)
            target[target_label] = 1.0
            total_error += np.mean((outputs - target) ** 2)
            correct += int(predicted_label == target_label)

        self.error = total_error / len(Agent.training_points)
        self.accuracy = correct / len(Agent.training_points)
        self.fitness = self.accuracy + 1.0 / (1.0 + self.error)

    def draw(self, interface, screen, generation):
        font = interface.font.SysFont('Consolas', 22)
        small_font = interface.font.SysFont('Consolas', 16)
        screen.blit(font.render('Voronoi partition learned by ENN', True, (20, 20, 20)), (32, 24))
        screen.blit(small_font.render(f'generation: {generation}', True, (20, 20, 20)), (36, 56))
        screen.blit(small_font.render(f'accuracy: {100 * self.accuracy:.2f}%', True, (20, 20, 20)), (220, 56))
        screen.blit(small_font.render(f'output MSE: {self.error:.6f}', True, (20, 20, 20)), (410, 56))

        left, top, size = 82, 96, 500
        pixels = 50
        cell = size // pixels
        for py in range(pixels):
            y = 1.0 - py / (pixels - 1)
            for px in range(pixels):
                x = px / (pixels - 1)
                label, _ = self.predict_label([x, y])
                rect = interface.Rect(left + px * cell, top + py * cell, cell + 1, cell + 1)
                interface.draw.rect(screen, Agent.colors[label], rect)

        interface.draw.rect(screen, (20, 20, 20), (left, top, size, size), 2)
        for centroid in Agent.centroids:
            cx = int(left + centroid[0] * size)
            cy = int(top + (1.0 - centroid[1]) * size)
            interface.draw.circle(screen, (0, 0, 0), (cx, cy), 7)

        screen.blit(small_font.render('Black dots are the five fixed centroids.', True, (20, 20, 20)), (left, top + size + 18))
