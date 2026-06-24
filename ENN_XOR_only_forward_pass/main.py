import numpy as np
import pygame as interface
from agent import Agent

class Main():
    interface.init()
    interface.display.set_caption('Neuro evolution')
    def __init__(self, number_of_agents):
        self.screen = interface.display.set_mode((750, 750))
        self.clock = interface.time.Clock()
        self.running = True
        self.agents = np.full(number_of_agents, None)
	
        weights = [-5.8460097, -5.6625414, -3.8732808, -3.804039, -7.8511896, 7.540539]
        biases = [2.1355941,  5.6442757, -3.4388447]
        
        for i in range(number_of_agents):
            self.agents[i] = Agent([i % 2, int(i % 4 > 1)])
            self.agents[i].neural_net.set_weights(weights)
            self.agents[i].neural_net.set_biases(biases)
	
    def run(self):	
        while self.running:
            for event in interface.event.get():
                if event.type == interface.QUIT:
                    self.running = False
                    interface.display.quit()
                    return

            self.update()				            
            self.draw()

    def update(self):
        for agent in self.agents:
           agent.update()         

    def draw(self):
        self.clock.tick(60)
        self.screen.fill(interface.Color('white'))
        
        # Draw agent(s)
        agent_to_draw = 1
        self.agents[agent_to_draw].draw(interface, self.screen) 
        
        # Update display
        interface.display.update()

# Create instance of Main class and run it
print('\014')
main = Main(4)
main.run()