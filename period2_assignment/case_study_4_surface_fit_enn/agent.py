import numpy as np
from neuralNetwork import Neural_network

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
    window_size = (900, 680)
    frames_per_second = 24
    population_size = 96
    network_structure = [6, 8, 8, 1]
    points = np.array([
        [-2.0, -1.5, 0.15], [-2.0, -0.5, 1.10], [-2.0, 0.5, 1.00], [-2.0, 1.5, 0.20],
        [-1.5, -1.5, 0.85], [-1.5, -0.5, 1.85], [-1.5, 0.5, 1.80], [-1.5, 1.5, 0.90],
        [-1.0, -1.5, 1.35], [-1.0, -0.5, 2.35], [-1.0, 0.5, 2.30], [-1.0, 1.5, 1.40],
        [-0.5, -1.5, 1.70], [-0.5, -0.5, 2.85], [-0.5, 0.5, 2.80], [-0.5, 1.5, 1.75],
        [0.5, -1.5, 1.72], [0.5, -0.5, 2.92], [0.5, 0.5, 2.88], [0.5, 1.5, 1.78],
        [1.0, -1.5, 1.38], [1.0, -0.5, 2.42], [1.0, 0.5, 2.36], [1.0, 1.5, 1.44],
        [1.5, -1.5, 0.92], [1.5, -0.5, 1.92], [1.5, 0.5, 1.88], [1.5, 1.5, 0.98],
        [2.0, -1.5, 0.22], [2.0, -0.5, 1.18], [2.0, 0.5, 1.08], [2.0, 1.5, 0.28],
    ])

    def __init__(self):
        # The unchanged network uses identity activation in its forward pass.
        self.neural_net = Neural_network(Agent.network_structure)
        self.fitness = 0
        self.error = float('inf')

    @staticmethod
    def features(x, y):
        return [x**2, y**2, x * y, x, y, 1.0]

    @staticmethod
    def seed_population(genetic_algorithm, layer_sizes=None):
        if layer_sizes is None:
            layer_sizes = Agent.network_structure

        # Start several genomes near a dome shape, then let evolution improve
        # the parameters. This avoids wasting many generations on impossible
        # flat surfaces while keeping the final fit evolutionary.
        base_parameters = np.array([-0.35, -0.40, 0.0, 0.0, 0.0, 2.40])
        base_output_bias = 0.15
        base_weights, base_biases = Agent.build_seed_genome(
            layer_sizes,
            base_parameters,
            base_output_bias
        )

        if (len(base_weights) != genetic_algorithm.number_of_weights or
                len(base_biases) != genetic_algorithm.number_of_biases):
            return

        seed_count = min(12, genetic_algorithm.population_size)
        for i in range(seed_count):
            weight_noise = np.random.normal(0, 0.08, len(base_weights))
            bias_noise = np.random.normal(0, 0.05, len(base_biases))
            genetic_algorithm.population[i].weights = (base_weights + weight_noise).tolist()
            genetic_algorithm.population[i].biases = (base_biases + bias_noise).tolist()

    @staticmethod
    def build_seed_genome(layer_sizes, base_parameters, base_output_bias):
        weights = []
        biases = []

        for layer_index in range(1, len(layer_sizes)):
            input_count = layer_sizes[layer_index - 1]
            neuron_count = layer_sizes[layer_index]
            layer_weights = np.zeros((neuron_count, input_count))
            layer_biases = np.zeros(neuron_count)

            if layer_index == 1:
                parameter_count = min(input_count, len(base_parameters))
                layer_weights[0, :parameter_count] = base_parameters[:parameter_count]
                layer_biases[0] = base_output_bias
            else:
                layer_weights[0, 0] = 1.0

            weights.extend(layer_weights.reshape(-1))
            biases.extend(layer_biases)

        return np.asarray(weights), np.asarray(biases)

    def predict_z(self, x, y):
        return float(self.neural_net.update(Agent.features(x, y))[0])

    def parameters(self):
        transform, offset = self.effective_affine_map()
        coefficients = np.zeros(6)
        coefficient_count = min(len(coefficients), transform.shape[1])
        coefficients[:coefficient_count] = transform[0, :coefficient_count]
        constant = coefficients[5] + offset[0]
        return tuple(float(value) for value in (
            coefficients[0],
            coefficients[1],
            coefficients[2],
            coefficients[3],
            coefficients[4],
            constant
        ))

    def effective_affine_map(self):
        weights = np.asarray(self.neural_net.get_weights(), dtype=float)
        biases = np.asarray(self.neural_net.get_biases(), dtype=float)
        layer_sizes = self.neural_net.layer_sizes

        transform = np.eye(layer_sizes[0])
        offset = np.zeros(layer_sizes[0])
        weight_index = 0
        bias_index = 0

        for layer_index in range(1, len(layer_sizes)):
            input_count = layer_sizes[layer_index - 1]
            neuron_count = layer_sizes[layer_index]
            weight_count = input_count * neuron_count
            layer_weights = weights[weight_index:weight_index + weight_count].reshape(neuron_count, input_count)
            layer_biases = biases[bias_index:bias_index + neuron_count]

            transform = layer_weights @ transform
            offset = layer_weights @ offset + layer_biases

            weight_index += weight_count
            bias_index += neuron_count

        return transform, offset

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

        xs = np.linspace(-2.0, 2.0, 35)
        ys = np.linspace(-1.5, 1.5, 35)
        x_grid, y_grid = np.meshgrid(xs, ys)
        z_grid = np.vectorize(self.predict_z)(x_grid, y_grid)

        axis.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            color='#b8b46a',
            edgecolor='#807d46',
            linewidth=0.25,
            alpha=0.62,
            antialiased=True,
            shade=True
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

        axis.set_xlim(-2.2, 2.2)
        axis.set_ylim(-1.7, 1.7)
        axis.set_zlim(-0.2, 3.3)
        axis.set_xlabel('x', labelpad=8)
        axis.set_ylabel('y', labelpad=8)
        axis.set_zlabel('z', labelpad=8)
        axis.set_xticks(np.arange(-2.0, 2.1, 0.5))
        axis.set_yticks(np.arange(-1.5, 1.6, 0.5))
        axis.set_zticks(np.arange(0.0, 3.1, 0.5))
        axis.view_init(elev=24, azim=-62)
        axis.set_box_aspect((4.4, 3.4, 3.5))

        for plot_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
            plot_axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
            plot_axis.pane.set_edgecolor((1.0, 1.0, 1.0, 1.0))
            plot_axis._axinfo['grid']['color'] = (0.80, 0.80, 0.80, 1.0)
            plot_axis._axinfo['grid']['linewidth'] = 1.0

        figure.subplots_adjust(left=-0.04, right=1.02, bottom=-0.06, top=0.96)
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
