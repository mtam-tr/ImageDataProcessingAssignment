import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from neuralNetwork import Neural_network
import sys
sys.dont_write_bytecode = True

class Agent():
    caption = 'Case 2 - ODE ENN'
    window_size = (820, 620)
    frames_per_second = 30
    population_size = 96
    domain = np.linspace(1.0, 1.5, 45) # input points

    def __init__(self):
        # one x value in, one y value out
        self.neural_net = Neural_network([1, 10, 1])
        self.fitness = 0
        self.error = float('inf')
        self.curve_error = float('inf')
        self._plot_cache_key = None
        self._plot_cache_surface = None

    @staticmethod
    def exact_solution(x): 
        return x**2 / 4.0 - x / 3.0 + 0.5 + 1.0 / (12.0 * x**2)

    def network_value(self, x): # give x to the neural network
        return self.neural_net.update([x])[0]

    def trial_solution(self, x):
        # force the boundary condition y(1)=0.5
        return 0.5 + (x - 1.0) * self.network_value(x)

    def derivative(self, x):  # Estimate dy/dx numerically
        h = 1e-3
        return (self.trial_solution(x + h) - self.trial_solution(x - h)) / (2.0 * h)

    def update(self):
        # measure the performance
        residuals = []
        curve_errors = []
        for x in Agent.domain:
            y = self.trial_solution(x)
            residual = x * self.derivative(x) + 2.0 * y - (x**2 - x + 1.0)
            residuals.append(residual**2)
            curve_errors.append((y - Agent.exact_solution(x)) ** 2)

        self.error = float(np.mean(residuals))
        self.curve_error = float(np.mean(curve_errors))
        # convert errors into a fitness score
        self.fitness = 1.0 / (1.0 + 20.0 * self.error + 2.0 * self.curve_error)

    def draw(self, interface, screen, generation):
        width, height = screen.get_size()
        cache_key = (width, height, generation, round(self.error, 12), round(self.curve_error, 12))
        if self._plot_cache_key == cache_key and self._plot_cache_surface is not None:
            screen.blit(self._plot_cache_surface, (0, 0))
            return

        x_min, x_max = 1.0, 1.5
        y_min, y_max = 0.49, 0.61
        xs = np.linspace(x_min, x_max, 120)
        exact_ys = [Agent.exact_solution(x) for x in xs]
        enn_ys = [self.trial_solution(x) for x in xs]
        sample_xs = Agent.domain[::3]
        sample_ys = [self.trial_solution(x) for x in sample_xs]

        figure = Figure(figsize=(width / 100, height / 100), dpi=100)
        figure.patch.set_facecolor('white')
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)

        axis.plot(xs, exact_ys, color='#1950e6', linewidth=2.0, label='exact solution')
        axis.plot(xs, enn_ys, color='#e62319', linewidth=2.0, label='ENN approximation')
        axis.scatter(sample_xs, sample_ys, s=22, color='#e62319', zorder=3)

        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.set_xticks(np.linspace(x_min, x_max, 6))
        axis.set_yticks(np.arange(0.50, 0.601, 0.02))
        axis.grid(True, color='#d4d4d4', linewidth=0.8)
        axis.set_title(r'$x\,dy/dx + 2y(x) = x^2 - x + 1$', pad=10)
        axis.legend(loc='lower left', bbox_to_anchor=(0.0, -0.28), ncol=2, frameon=False)

        for spine in axis.spines.values():
            spine.set_color('#5a5a5a')
            spine.set_linewidth(1.0)

        figure.subplots_adjust(left=0.12, right=0.96, bottom=0.23, top=0.82)
        figure.text(
            0.04,
            0.955,
            f'generation: {generation}    residual MSE: {self.error:.8f}',
            ha='left',
            va='top',
            fontsize=10,
            family='monospace',
            color='black'
        )

        canvas.draw()
        size = canvas.get_width_height()
        image = interface.image.frombuffer(canvas.buffer_rgba().tobytes(), size, 'RGBA').copy()
        self._plot_cache_key = cache_key
        self._plot_cache_surface = image
        screen.blit(image, (0, 0))
