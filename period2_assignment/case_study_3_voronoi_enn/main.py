import copy
import os
import sys

if '--smoke-test' in sys.argv:
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import pygame as interface

from agent import Agent
from geneticAlgorithm import Genetic_algorithm


class Main():
    interface.init()

    def __init__(self, number_of_agents):
        self.screen = interface.display.set_mode(Agent.window_size)
        interface.display.set_caption(Agent.caption)
        self.clock = interface.time.Clock()
        self.running = True
        self.generation = 0
        self.best_agent = None
        self.agents = [Agent() for _ in range(number_of_agents)]

        number_of_weights = self.agents[0].neural_net.get_number_of_weights()
        number_of_biases = self.agents[0].neural_net.get_number_of_biases()
        self.genetic_algorithm = Genetic_algorithm(number_of_agents, number_of_weights, number_of_biases)

        if hasattr(Agent, 'seed_population'):
            Agent.seed_population(self.genetic_algorithm)

        self.apply_genomes_to_agents()

    def apply_genomes_to_agents(self):
        for i, agent in enumerate(self.agents):
            agent.neural_net.set_weights(self.genetic_algorithm.population[i].weights)
            agent.neural_net.set_biases(self.genetic_algorithm.population[i].biases)

    def run(self, max_generations=None):
        while self.running:
            for event in interface.event.get():
                if event.type == interface.QUIT:
                    self.running = False

            self.update()
            self.draw()

            if max_generations is not None and self.generation >= max_generations:
                self.running = False

        interface.display.quit()
        if self.best_agent is not None:
            print(
                f'{Agent.caption}: generations={self.generation}, '
                f'best_fitness={self.best_agent.fitness:.6f}, '
                f'error={self.best_agent.error:.6f}'
            )

    def update(self):
        for agent in self.agents:
            agent.update()

        self.genetic_algorithm.update(self.agents)
        generation_best = max(self.agents, key=lambda agent: agent.fitness)
        if self.best_agent is None or generation_best.fitness > self.best_agent.fitness:
            self.best_agent = copy.deepcopy(generation_best)

        self.genetic_algorithm.upgrade()
        self.generation += 1
        self.apply_genomes_to_agents()

    def draw(self):
        self.clock.tick(Agent.frames_per_second)
        self.screen.fill(interface.Color('white'))

        if self.best_agent is not None:
            self.best_agent.draw(interface, self.screen, self.generation)

        interface.display.flip()


print('\014')
main = Main(Agent.population_size)
main.run(max_generations=8 if '--smoke-test' in sys.argv else 100)
