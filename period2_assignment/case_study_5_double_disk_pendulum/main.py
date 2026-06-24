import os
import sys

if '--smoke-test' in sys.argv:
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import pygame as interface
import pymunk
import pymunk.pygame_util

from agent import Agent
from geneticAlgorithm import Genetic_algorithm


class Main():
    interface.init()

    def __init__(self, number_of_agents):
        self.screen = interface.display.set_mode((720, 620), interface.DOUBLEBUF)
        interface.display.set_caption('Case 5 - Double Disk Pendulum ENN')
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        pymunk.pygame_util.positive_y_is_up = True
        self.clock = interface.time.Clock()
        self.running = True
        self.dead_agents = 0
        self.generation = 0
        self.best_fitness = 0
        self.world = pymunk.Space()
        self.world.gravity = (0.0, -981.0)

        position_of_agent = (0.5 * interface.display.get_window_size()[0], 120)
        self.agents = [Agent(position_of_agent, self.world) for _ in range(number_of_agents)]

        number_of_weights = self.agents[0].neural_net.get_number_of_weights()
        number_of_biases = self.agents[0].neural_net.get_number_of_biases()
        self.genetic_algorithm = Genetic_algorithm(number_of_agents, number_of_weights, number_of_biases)
        self.apply_genomes_to_agents()

    def apply_genomes_to_agents(self):
        for i, agent in enumerate(self.agents):
            agent.neural_net.set_weights(self.genetic_algorithm.population[i].weights)
            agent.neural_net.set_biases(self.genetic_algorithm.population[i].biases)

    def run(self, max_steps=None):
        steps = 0
        while self.running:
            for event in interface.event.get():
                if event.type == interface.QUIT:
                    self.running = False
                if event.type == interface.MOUSEBUTTONDOWN:
                    for agent in self.agents:
                        if agent.is_alive:
                            agent.is_alive = False
                            agent.destroy()
                    self.dead_agents = len(self.agents)

            self.update()
            self.draw()
            steps += 1

            if max_steps is not None and steps >= max_steps:
                self.running = False

        self.genetic_algorithm.update(self.agents)
        self.best_fitness = max(self.best_fitness, max(agent.fitness for agent in self.agents))
        interface.display.quit()
        print(
            f'Case 5 - Double Disk Pendulum ENN: '
            f'generation={self.generation}, best_fitness={self.best_fitness:.3f}'
        )

    def update(self):
        for agent in self.agents:
            if agent.update():
                self.dead_agents += 1

        if self.dead_agents == len(self.agents):
            self.genetic_algorithm.update(self.agents)
            self.best_fitness = max(self.best_fitness, max(agent.fitness for agent in self.agents))
            self.genetic_algorithm.upgrade()
            self.dead_agents = 0
            self.generation += 1
            for agent in self.agents:
                agent.reset()
            self.apply_genomes_to_agents()

        self.world.step(0.005)

    def draw(self):
        self.clock.tick(240)
        self.screen.fill(interface.Color('white'))
        font = interface.font.SysFont('Consolas', 18)
        self.screen.blit(font.render(f'generation: {self.generation}', True, (20, 20, 20)), (24, 18))
        self.screen.blit(font.render(f'best fitness: {self.best_fitness:.1f}', True, (20, 20, 20)), (220, 18))
        self.screen.blit(font.render('click to force next generation', True, (20, 20, 20)), (440, 18))

        drawn = 0
        for agent in self.agents:
            if agent.is_alive:
                agent.draw(interface, self.screen)
                drawn += 1
                if drawn >= 4:
                    break
        Agent.draw_shapes([Agent.shape_plateau], interface, self.screen)
        interface.display.flip()


print('\014')
main = Main(48)
main.run(max_steps=240 if '--smoke-test' in sys.argv else None)
