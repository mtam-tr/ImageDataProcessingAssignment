import pygame as interface
from agent import Agent

class Main:
    interface.init()
    interface.display.set_caption('Forward pass')

    def __init__(self):
        self.size = 720
        self.cell = 16

        self.screen = interface.display.set_mode((self.size, self.size))

        self.clock = interface.time.Clock()
        self.running = True

        self.agent = Agent()

    def run(self):
        while self.running:
            for event in interface.event.get():
                if event.type == interface.QUIT:
                    self.running = False                    
                
                if event.type == interface.KEYDOWN:
                    self.agent.next_preset()
    
            # Mouse coords
            self.agent.update()
    
            # Draw via agent
            self.agent.draw(interface, self.screen, self.size, self.cell)
    
            interface.display.set_caption(
                f'{self.agent.activation_functions[1]}'
            )
    
            interface.display.flip()
            self.clock.tick(60)
    
        interface.quit()

print('\014')
main = Main()
main.run()