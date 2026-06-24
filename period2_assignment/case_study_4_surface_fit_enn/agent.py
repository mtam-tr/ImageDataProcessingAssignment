import numpy as np
from neuralNetwork import Neural_network

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required by assignment text.
except Exception:
    Axes3D = None


class Agent():
    caption = 'Case 4 - Quadratic Surface ENN'
    window_size = (900, 680)
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
    ])

    def __init__(self):
        # Six engineered inputs make the ENN output exactly the quadratic form:
        # z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f.
        self.neural_net = Neural_network([6, 1])
        self.activation_functions = [None, 'id']
        self.fitness = 0
        self.error = float('inf')

    @staticmethod
    def features(x, y):
        return [x**2, y**2, x * y, x, y, 1.0]

    @staticmethod
    def seed_population(genetic_algorithm):
        # Start several genomes near a dome shape, then let evolution improve
        # the parameters. This avoids wasting many generations on impossible
        # flat surfaces while keeping the final fit evolutionary.
        base_weights = [-0.35, -0.40, 0.0, 0.0, 0.0, 2.40]
        base_biases = [0.15]
        seed_count = min(12, genetic_algorithm.population_size)
        for i in range(seed_count):
            noise = np.random.normal(0, 0.25, len(base_weights))
            bias_noise = np.random.normal(0, 0.25, len(base_biases))
            genetic_algorithm.population[i].weights = (np.asarray(base_weights) + noise).tolist()
            genetic_algorithm.population[i].biases = (np.asarray(base_biases) + bias_noise).tolist()

    def predict_z(self, x, y):
        return self.neural_net.update(Agent.features(x, y), self.activation_functions)[0]

    def parameters(self):
        weights = self.neural_net.get_weights()
        bias = self.neural_net.get_biases()[0]
        return weights[0], weights[1], weights[2], weights[3], weights[4], weights[5] + bias

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

    def draw(self, interface, screen, generation):
        font = interface.font.SysFont('Consolas', 22)
        small_font = interface.font.SysFont('Consolas', 15)
        screen.blit(font.render('Quadratic surface fitted by ENN', True, (20, 20, 20)), (32, 24))
        screen.blit(small_font.render(f'generation: {generation}', True, (20, 20, 20)), (36, 56))
        screen.blit(small_font.render(f'fitting MSE: {self.error:.6f}', True, (20, 20, 20)), (220, 56))
        if Axes3D is None:
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

        a, b, c, d, e, f = self.parameters()
        text = f'z = {a:+.3f}x^2 {b:+.3f}y^2 {c:+.3f}xy {d:+.3f}x {e:+.3f}y {f:+.3f}'
        screen.blit(small_font.render(text, True, (20, 20, 20)), (36, 625))
        screen.blit(small_font.render('red: measured points    blue: evolved surface', True, (20, 20, 20)), (36, 602))
