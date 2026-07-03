import sys
sys.dont_write_bytecode = True
import pygame as interface
import pymunk
import pymunk.pygame_util
from agent import Agent
from geneticAlgorithm import Genetic_algorithm 


class Main():
    interface.init()
    def __init__(self, number_of_agents):        
        # Create the Pygame window and configure Pymunk debug drawing.
        self.screen = interface.display.set_mode((600, 600), interface.DOUBLEBUF)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        self.draw_options.flags |= pymunk.SpaceDebugDrawOptions.DRAW_CONSTRAINTS
        pymunk.pygame_util.positive_y_is_up = True
        self.clock = interface.time.Clock()
        self.running = True
        self.dead_agents = 0
        self.agents = [None] * number_of_agents        
        self.world = pymunk.Space()
        self.world.gravity = (0.0, -981.0)
        
        # Create all cart-pole agents in the same physics world.
        position_of_agent = (0.5 * interface.display.get_window_size()[0], 150)
        for i in range(0, number_of_agents):
            self.agents[i] = Agent(position_of_agent, self.world)	
        Agent.is_initialized = False
            
        # The genetic algorithm needs to know how many genes each neural net has.
        number_of_weights = self.agents[0].neural_net.get_number_of_weights()
        number_of_biases = self.agents[0].neural_net.get_number_of_biases()
        self.genetic_algorithm = Genetic_algorithm(number_of_agents, number_of_weights, number_of_biases)
		
        # Give every agent one genome: a flat list of weights and a flat list of biases.
        for i in range(number_of_agents):
            self.agents[i].neural_net.set_weights(self.genetic_algorithm.population[i].weights)
            self.agents[i].neural_net.set_biases(self.genetic_algorithm.population[i].biases)
	
    def run(self):	
        while self.running:
            for event in interface.event.get():
                if event.type == interface.QUIT:
                    self.running = False
                    interface.display.quit()
                    return
                if event.type == interface.MOUSEBUTTONDOWN:
                    # Mouse click is a shortcut to end the current generation immediately.
                    print('Forcing a new generation')
                    for agent in self.agents:
                        agent.is_alive = False
                        agent.destroy()
                        self.dead_agents = len(self.agents)
                        
            self.update()	
            self.draw()

    def update(self):
        # agent.update() returns True only when that agent just died.
        for agent in self.agents:
            if agent.update():
                self.dead_agents += 1
        
        # When the full population is dead, score it and create a new generation.
        if(self.dead_agents == len(self.agents)):
            self.genetic_algorithm.update(self.agents)
            self.genetic_algorithm.upgrade()
            self.dead_agents = 0
            for i, agent in enumerate(self.agents):
                agent.reset()
                agent.neural_net.set_weights(self.genetic_algorithm.population[i].weights)
                agent.neural_net.set_biases(self.genetic_algorithm.population[i].biases)
                        
        # Advance the physics simulation by a small fixed time step.
        self.world.step(0.005)        

    def draw(self):
        self.clock.tick(500)
        interface.display.set_caption(f'FPS: {self.clock.get_fps() :.0f}')

        self.screen.fill(interface.Color('white'))      
        
        # self.world.debug_draw(self.draw_options)
        
        # Draw the first (best) four agents
        i = 0
        for agent in self.agents:
            if not agent.is_alive:
                continue
            agent.draw(interface, self.screen)
            i += 1
            if i > 3:
                break
        # Draw plateau
        Agent.draw_shapes([Agent.shape_plateau], interface, self.screen) 
        
        interface.display.flip()

# Create instance of Main class and run it
print('\014')
main = Main(2**6)
main.run()
