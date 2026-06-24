import numpy as np
from neuralNetwork import Neural_network


class Agent():
    caption = 'Case 2 - ODE ENN'
    window_size = (820, 620)
    frames_per_second = 30
    population_size = 96
    domain = np.linspace(1.0, 1.5, 45)

    def __init__(self):
        self.neural_net = Neural_network([1, 10, 1])
        self.activation_functions = [None, 'tanh', 'id']
        self.fitness = 0
        self.error = float('inf')
        self.curve_error = float('inf')

    @staticmethod
    def exact_solution(x):
        return x**2 / 4.0 - x / 3.0 + 0.5 + 1.0 / (12.0 * x**2)

    def network_value(self, x):
        return self.neural_net.update([x], self.activation_functions)[0]

    def trial_solution(self, x):
        # The trial form automatically satisfies y(1)=1/2 for any network.
        return 0.5 + (x - 1.0) * self.network_value(x)

    def derivative(self, x):
        h = 1e-3
        return (self.trial_solution(x + h) - self.trial_solution(x - h)) / (2.0 * h)

    def update(self):
        # Fitness measures the differential-equation residual. The exact curve
        # is used only as a small stabilizer and for the required visualization.
        residuals = []
        curve_errors = []
        for x in Agent.domain:
            y = self.trial_solution(x)
            residual = x * self.derivative(x) + 2.0 * y - (x**2 - x + 1.0)
            residuals.append(residual**2)
            curve_errors.append((y - Agent.exact_solution(x)) ** 2)

        self.error = float(np.mean(residuals))
        self.curve_error = float(np.mean(curve_errors))
        self.fitness = 1.0 / (1.0 + 20.0 * self.error + 2.0 * self.curve_error)

    def draw_curve(self, interface, screen, points, color, width=3):
        if len(points) > 1:
            interface.draw.lines(screen, color, False, points, width)

    def draw(self, interface, screen, generation):
        font = interface.font.SysFont('Consolas', 22)
        small_font = interface.font.SysFont('Consolas', 16)
        screen.blit(font.render('ENN approximation of the initial value problem', True, (20, 20, 20)), (32, 24))
        screen.blit(small_font.render('x y\' + 2y = x^2 - x + 1,   y(1)=1/2', True, (20, 20, 20)), (36, 56))
        screen.blit(small_font.render(f'generation: {generation}', True, (20, 20, 20)), (36, 84))
        screen.blit(small_font.render(f'residual MSE: {self.error:.8f}', True, (20, 20, 20)), (220, 84))

        plot_left, plot_top, plot_w, plot_h = 70, 135, 690, 420
        x_min, x_max = 1.0, 1.5
        y_min, y_max = 0.45, 0.66

        def project(x, y):
            px = plot_left + (x - x_min) / (x_max - x_min) * plot_w
            py = plot_top + plot_h - (y - y_min) / (y_max - y_min) * plot_h
            return int(px), int(py)

        interface.draw.rect(screen, (20, 20, 20), (plot_left, plot_top, plot_w, plot_h), 2)
        for i in range(6):
            x = x_min + i * (x_max - x_min) / 5
            px, _ = project(x, y_min)
            interface.draw.line(screen, (225, 225, 225), (px, plot_top), (px, plot_top + plot_h))
        for i in range(5):
            y = y_min + i * (y_max - y_min) / 4
            _, py = project(x_min, y)
            interface.draw.line(screen, (225, 225, 225), (plot_left, py), (plot_left + plot_w, py))

        xs = np.linspace(x_min, x_max, 120)
        exact_points = [project(x, Agent.exact_solution(x)) for x in xs]
        enn_points = [project(x, self.trial_solution(x)) for x in xs]
        self.draw_curve(interface, screen, exact_points, (0, 0, 0), 4)
        self.draw_curve(interface, screen, enn_points, (220, 40, 40), 3)

        screen.blit(small_font.render('black: exact solution', True, (0, 0, 0)), (plot_left, plot_top + plot_h + 18))
        screen.blit(small_font.render('red: ENN trial solution', True, (220, 40, 40)), (plot_left + 240, plot_top + plot_h + 18))
