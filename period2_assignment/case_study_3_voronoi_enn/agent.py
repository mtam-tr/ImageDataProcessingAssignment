import numpy as np
from neuralNetwork import Neural_network
import sys
sys.dont_write_bytecode = True

class Agent():
    caption = 'Case 3 - Voronoi ENN'
    window_size = (760, 650)
    frames_per_second = 18
    population_size = 40
    # fixed points that define the voronoi regions
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
    # grid points used for training.
    training_points = np.array([[x, y] for y in np.linspace(0, 1, 20) for x in np.linspace(0, 1, 20)])

    def __init__(self):
        # x/y coordinates in, one score for each centroid out.
        self.neural_net = Neural_network([2, 5, 5])
        self.fitness = 0
        self.error = float('inf')
        self.accuracy = 0

    @staticmethod
    def nearest_centroid(point):
        # find the closest centroid for this point.
        distances = np.sum((Agent.centroids - point) ** 2, axis=1)
        return int(np.argmin(distances))

    def predict_label(self, point):
        # choose the centroid with the highest network output.
        outputs = self.neural_net.update(point)
        return int(np.argmax(outputs)), np.asarray(outputs, dtype=float)

    def update(self):
        # measure how well the network predicts the voronoi regions.
        correct = 0
        total_error = 0.0

        for point in Agent.training_points:
            target_label = Agent.nearest_centroid(point)
            predicted_label, outputs = self.predict_label(point)

            correct += int(predicted_label == target_label)

            # reward a higher score for the correct centroid.
            target_score = outputs[target_label]
            wrong_scores = np.delete(outputs, target_label)
            best_wrong_score = np.max(wrong_scores)

            margin = target_score - best_wrong_score
            total_error += max(0.0, 1.0 - margin)

        self.accuracy = correct / len(Agent.training_points)
        self.error = total_error / len(Agent.training_points)
        # convert accuracy and error into a fitness score.
        self.fitness = self.accuracy + 1.0 / (1.0 + self.error)

    def draw(self, interface, screen, generation):
        font = interface.font.SysFont('Consolas', 22)
        small_font = interface.font.SysFont('Consolas', 16)
        screen.blit(font.render('Voronoi partition learned by ENN', True, (20, 20, 20)), (32, 24))
        screen.blit(small_font.render(f'generation: {generation}', True, (20, 20, 20)), (36, 56))
        screen.blit(small_font.render(f'accuracy: {100 * self.accuracy:.2f}%', True, (20, 20, 20)), (220, 56))
        screen.blit(small_font.render(f'output error: {self.error:.6f}', True, (20, 20, 20)), (410, 56))

        left, top, size = 82, 96, 500
        pixels = 50
        cell = size // pixels
        # draw the learned voronoi regions.
        for py in range(pixels):
            y = 1.0 - py / (pixels - 1)
            for px in range(pixels):
                x = px / (pixels - 1)
                label, _ = self.predict_label([x, y])
                rect = interface.Rect(left + px * cell, top + py * cell, cell + 1, cell + 1)
                interface.draw.rect(screen, Agent.colors[label], rect)

        interface.draw.rect(screen, (20, 20, 20), (left, top, size, size), 2)
        # draw the centroid locations.
        for centroid in Agent.centroids:
            cx = int(left + centroid[0] * size)
            cy = int(top + (1.0 - centroid[1]) * size)
            interface.draw.circle(screen, (0, 0, 0), (cx, cy), 7)
