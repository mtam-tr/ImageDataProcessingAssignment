import sys
sys.dont_write_bytecode = True

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from neuralNetwork import Neural_network


class Agent:
    caption = 'Case 4 - Quadratic Surface ENN'
    window_size = (960, 740)
    frames_per_second = 24
    population_size = 96
    points = np.array([
        [-2.0, -1.5, 0.15], [-2.0, -0.5, 1.10], [-2.0, 0.5, 1.00], [-2.0, 1.5, 0.20],
        [-1.5, -1.5, 0.85], [-1.5, -0.5, 1.85], [-1.5, 0.5, 1.80], [-1.5, 1.5, 0.90],
        [-1.0, -1.5, 1.35], [-1.0, -0.5, 2.35], [-1.0, 0.5, 2.30], [-1.0, 1.5, 1.40],
        [-0.5, -1.5, 1.70], [-0.5, -0.5, 2.85], [-0.5, 0.5, 2.80], [-0.5, 1.5, 1.75],
        [0.5, -1.5, 1.72], [0.5, -0.5, 2.92], [0.5, 0.5, 2.88], [0.5, 1.5, 1.78],
        [1.0, -1.5, 1.38], [1.0, -0.5, 2.42], [1.0, 0.5, 2.36], [1.0, 1.5, 1.44],
        [1.5, -1.5, 0.92], [1.5, -0.5, 1.92], [1.5, 0.5, 1.88], [1.5, 1.5, 0.98],
        [2.0, -1.5, 0.22], [2.0, -0.5, 1.18], [2.0, 0.5, 1.08], [2.0, 1.5, 0.28],
    ]) # GRID OF POINTS

    def __init__(self):
        # The unchanged network uses identity activation in its forward pass.
        self.neural_net = Neural_network([6, 1])
        self.fitness = 0
        self.error = float('inf')

    @staticmethod
    def features(x, y): # TURN 2 INPUTS INTO 6 FEATURES
        return [x**2, y**2, x * y, x, y, 1.0]

    def predict_z(self, x, y):
        return float(self.neural_net.update(Agent.features(x, y))[0])

    def update(self):
        # Fitness is the inverse of the mean squared fitting error over the
        # provided 32 measured 3D points.
        errors = [(self.predict_z(x, y) - z) ** 2 for x, y, z in Agent.points]
        self.error = float(np.mean(errors))
        self.fitness = 1.0 / (1.0 + 25.0 * self.error)

    def draw(self, interface, screen, generation):
        width, height = screen.get_size()
        figure = Figure(figsize=(width / 100, height / 100), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111, projection='3d')

        xs = np.linspace(-2.0, 2.0, 80)
        ys = np.linspace(-1.5, 1.5, 80)
        x_grid, y_grid = np.meshgrid(xs, ys)
        z_grid = np.vectorize(self.predict_z)(x_grid, y_grid) 

        axis.plot_surface(x_grid, y_grid, z_grid, color='#b8b46a', alpha=0.6)
        axis.scatter(
            Agent.points[:, 0],
            Agent.points[:, 1],
            Agent.points[:, 2],
            s=38,
            c='#ff2318',
        )

        x_limits = (-2.15, 2.15)
        y_limits = (-1.65, 1.65)
        z_limits = (-0.2, 3.35)
        axis.set(
            xlim=x_limits,
            ylim=y_limits,
            zlim=z_limits,
            xlabel='x',
            ylabel='y',
            zlabel='z',
        )
        axis.set_xticks(np.arange(-2.0, 2.1, 0.5))
        axis.set_yticks(np.arange(-1.5, 1.6, 0.5))
        axis.set_zticks(np.arange(0.0, 3.1, 0.5))
        axis.view_init(elev=24, azim=-62)
        axis.set_box_aspect((4.3, 3.3, 3.55))

        figure.subplots_adjust(left=0.02, right=0.94, bottom=0.08, top=0.94)
        figure.text(
            0.03,
            0.965,
            f'generation: {generation}    fitting MSE: {self.error:.6f}',
            ha='left',
            va='top',
            fontsize=10,
            family='monospace',
            color='black'
        )

        canvas.draw()
        size = canvas.get_width_height()
        image = interface.image.frombuffer(canvas.buffer_rgba().tobytes(), size, 'RGBA')
        screen.blit(image, (0, 0))
