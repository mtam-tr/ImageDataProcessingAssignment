import numpy as np
from neuralNetwork import Neural_network
import sys
sys.dont_write_bytecode = True
try:
    from mpl_toolkits.mplot3d import Axes3D 
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
except Exception:
    Axes3D = None
    FigureCanvasAgg = None
    Figure = None


class Agent():
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
        self.neural_net = Neural_network([6,1])
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
        errors = []
        for x, y, z in Agent.points:
            errors.append((self.predict_z(x, y) - z) ** 2)
        self.error = float(np.mean(errors))
        self.fitness = 1.0 / (1.0 + 25.0 * self.error)

    def project(self, x, y, z):
        center_x, center_y = 450, 430
        scale = 95
        px = center_x + scale * 0.78 * (x - y)
        py = center_y - scale * (0.38 * (x + y) + 0.72 * z)
        return int(px), int(py)

    def draw_matplotlib_surface(self, interface, screen, generation):
        width, height = screen.get_size()
        figure = Figure(figsize=(width / 100, height / 100), dpi=100)
        figure.patch.set_facecolor('white')
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111, projection='3d')

        xs = np.linspace(-2.0, 2.0, 80)
        ys = np.linspace(-1.5, 1.5, 80)
        x_grid, y_grid = np.meshgrid(xs, ys)
        z_grid = np.vectorize(self.predict_z)(x_grid, y_grid) 

        axis.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            color='#b8b46a',
            edgecolor='none',
            linewidth=0,
            alpha=0.6,
            antialiased=False,
            shade=False
        )
        axis.scatter(
            Agent.points[:, 0],
            Agent.points[:, 1],
            Agent.points[:, 2],
            s=38,
            c='#ff2318',
            edgecolors='#6e1713',
            linewidths=0.9,
            depthshade=True
        )

        x_limits = (-2.15, 2.15)
        y_limits = (-1.65, 1.65)
        z_limits = (-0.2, 3.35)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_zlim(*z_limits)
        axis.set_xlabel('x', labelpad=8)
        axis.set_ylabel('y', labelpad=8)
        axis.set_zlabel('z', labelpad=8)
        axis.set_xticks(np.arange(-2.0, 2.1, 0.5))
        axis.set_yticks(np.arange(-1.5, 1.6, 0.5))
        axis.set_zticks(np.arange(0.0, 3.1, 0.5))
        axis.view_init(elev=24, azim=-62)
        axis.set_box_aspect((
            x_limits[1] - x_limits[0],
            y_limits[1] - y_limits[0],
            z_limits[1] - z_limits[0]
        ))
        axis.grid(True)

        for plot_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
            plot_axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
            plot_axis.pane.set_edgecolor((1.0, 1.0, 1.0, 1.0))
            plot_axis._axinfo['grid']['color'] = (0.80, 0.80, 0.80, 1.0)
            plot_axis._axinfo['grid']['linewidth'] = 1.0

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

    def draw_pygame_fallback(self, interface, screen, generation):
        font = interface.font.SysFont('Consolas', 22)
        small_font = interface.font.SysFont('Consolas', 15)
        screen.blit(font.render('Quadratic surface fitted by ENN', True, (20, 20, 20)), (32, 24))
        screen.blit(small_font.render(f'generation: {generation}', True, (20, 20, 20)), (36, 56))
        screen.blit(small_font.render(f'fitting MSE: {self.error:.6f}', True, (20, 20, 20)), (220, 56))
        screen.blit(small_font.render('Matplotlib not installed: using pygame 3D projection fallback.', True, (130, 90, 0)), (36, 80))

        xs = np.linspace(-2.0, 2.0, 17)
        ys = np.linspace(-1.5, 1.5, 17)

        for y in ys:
            line = [self.project(x, y, self.predict_z(x, y)) for x in xs]
            interface.draw.lines(screen, (60, 130, 220), False, line, 1)
        for x in xs:
            line = [self.project(x, y, self.predict_z(x, y)) for y in ys]
            interface.draw.lines(screen, (60, 130, 220), False, line, 1)

        for x, y, z in Agent.points:
            interface.draw.circle(screen, (220, 40, 40), self.project(x, y, z), 5)

    def draw(self, interface, screen, generation):
        if Axes3D is None or FigureCanvasAgg is None or Figure is None:
            self.draw_pygame_fallback(interface, screen, generation)
            return

        self.draw_matplotlib_surface(interface, screen, generation)
