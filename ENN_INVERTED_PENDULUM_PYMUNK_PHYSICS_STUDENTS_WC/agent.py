import numpy as np
import pymunk
from neuralNetwork import Neural_network

class Agent():	
    # Static fields
    counter         = -1  
    is_initialized = False
    time_no_disturb = 2000
    
    # Collision categories/masks decide which Pymunk shapes can collide.
    category_cart   = 0b0001
    mask_cart       = 0b1000
    category_pole   = 0b0010
    mask_pole       = 0b0000
    category_bob    = 0b0100
    mask_bob        = 0b0000
    
    # Convenience function to transform the (x, +y) system to Pygame's (x, -y) coordinate system        
    @staticmethod
    def x2x_y2miny(v, interface):
        return v.x, interface.display.get_window_size()[1] - v.y
    # Draw pymunk shapes with pygame    
    @staticmethod
    def draw_shapes(shapes, interface, screen):
        for shape in shapes: 
            v = shape.body.position
            if type(shape) == pymunk.shapes.Poly:
                V = [v_poly.rotated(shape.body.angle) + v for v_poly in shape.get_vertices()]              
                V = list(map(lambda v: Agent.x2x_y2miny(v, interface), V))                
                interface.draw.polygon(screen, shape.color, V, 0)
                interface.draw.aalines(screen, interface.Color('black'), True, V)
            elif type(shape) == pymunk.shapes.Segment:
                rotation_vector = shape.body.rotation_vector                           
                v_1 = Agent.x2x_y2miny(v + shape.a.cpvrotate(rotation_vector), interface)
                v_2 = Agent.x2x_y2miny(v + shape.b.cpvrotate(rotation_vector), interface)
                thickness = int(2 * shape.radius)
                interface.draw.line(screen, shape.color, v_1, v_2, thickness)
            elif type(shape) == pymunk.shapes.Circle:
                rotation_vector = shape.body.rotation_vector   
                v_1 = Agent.x2x_y2miny(v, interface) 
                v_2 = Agent.x2x_y2miny(v + shape.radius * rotation_vector, interface) 
                interface.draw.circle(screen, shape.color, v_1, shape.radius)
                interface.draw.circle(screen, interface.Color('black'), v_1, shape.radius, 1)
                interface.draw.aaline(screen, interface.Color('black'), v_1, v_2)
    # Turn off collisions between distinct agents    
    @staticmethod
    def collision_callback(arbiter, world, data):
        collision_indices = []
        for shape in arbiter.shapes:
            collision_indices.append(shape.pair_index)
        # Only allow collision when both shapes belong to the same agent.
        return len(np.unique(collision_indices)) == 1    
    
    # Constructor
    def __init__(self, position_of_agent, world):
        # Statics
        Agent.counter += 1
        
        if not Agent.is_initialized:
            # The plateau is shared by all agents, so it is created only once.
            Agent.width_plateau = 500
            Agent.height_plateau = 15
            Agent.body_plateau = pymunk.Body(body_type = pymunk.Body.STATIC)    
            Agent.body_plateau.position = (position_of_agent[0], position_of_agent[1] - 0.5 * Agent.height_plateau)
            Agent.shape_plateau = pymunk.Poly.create_box(Agent.body_plateau, (Agent.width_plateau, Agent.height_plateau))
            Agent.shape_plateau.friction = 0
            Agent.shape_plateau.color = (0, 255, 0, 255)
            world.add(Agent.body_plateau, Agent.shape_plateau)            
            # Set collision handler
            world.on_collision(1, 1, begin = Agent.collision_callback)
            
            Agent.is_initialized = True
            
        # Local geometry/parameters of agent
        self.id                = Agent.counter
        self.position_of_agent = position_of_agent     
        self.width_cart        = 75
        self.height_cart       = 30
        self.mass_cart         = 1
        self.height_pole       = 250
        self.width_pole        = 20
        self.mass_pole         = 1
        self.radius_bob        = 20
        self.mass_bob          = 1
        self.max_force         = 50
        self.angle_treshold    = np.radians(15)
        # Controller network: pole angle, pole angular speed, cart position, cart speed -> force.
        self.neural_net        = Neural_network([4, 8, 1])
        self.inputs            = [None] * self.neural_net.number_of_inputs        
        self.world             = world
        self.reset()

    def reset(self):
        # Reset one candidate controller for a new generation attempt.
        self.is_alive        = True
        self.fitness         = 0
        self.time_no_disturb = Agent.time_no_disturb     
        self.shapes          = []
        self.joints          = []
        self.create_agent()

    def create_agent(self):
        # Cart: horizontal moving base constrained to slide along the plateau.
        body_cart = pymunk.Body()
        body_cart.position = (self.position_of_agent[0], self.position_of_agent[1] + 0.5 * self.height_cart)
        shape_cart = pymunk.Poly.create_box(body_cart, (self.width_cart, self.height_cart))
        shape_cart.filter = pymunk.ShapeFilter(categories = Agent.category_cart, mask = Agent.mask_cart)
        shape_cart.mass = self.mass_cart
        shape_cart.friction = 0
        shape_cart.color = (0, 0, 255, 255)
        prismatic_joint_cart = pymunk.GrooveJoint(Agent.body_plateau, body_cart, 
                                                  (-0.5 * Agent.width_plateau, 0.5 * Agent.height_plateau),                                                                                  
                                                  ( 0.5 * Agent.width_plateau, 0.5 * Agent.height_plateau), (0, -0.5 * self.height_cart))
                
        # Pole: hinged to the cart, so it can rotate and fall.
        body_pole = pymunk.Body()
        body_pole.position = (body_cart.position.x, body_cart.position.y + 0.5 * self.height_pole)
        shape_pole = pymunk.Segment(body_pole, (0, -0.5 * self.height_pole), (0, 0.5 * self.height_pole), 0.5 * self.width_pole)
        shape_pole.filter = pymunk.ShapeFilter(categories = Agent.category_pole, mask = Agent.mask_pole)
        shape_pole.mass = self.mass_pole
        shape_pole.color = (255, 255, 0, 0)
        revolute_joint_pole = pymunk.PivotJoint(body_pole, body_cart, (0, -0.5 * self.height_pole), (0, 0))

        # Bob: weight at the top of the pole.
        body_bob = pymunk.Body()
        body_bob.position = (body_pole.position.x, body_pole.position.y + 0.5 * self.height_pole)
        shape_bob = pymunk.Circle(body_bob, self.radius_bob)
        shape_bob.filter = pymunk.ShapeFilter(categories = Agent.category_bob, mask = Agent.mask_bob)
        shape_bob.mass = self.mass_bob
        shape_bob.color = (255, 0, 0, 255)        
        pin_joint_bob = pymunk.PinJoint(body_pole, body_bob, (0, 0.5 * self.height_pole), (0, 0))
        rotary_limit_joint_bob = pymunk.RotaryLimitJoint(body_pole, body_bob, 0, 0)
        
        # Add bodies, shapes and joints to the world
        self.world.add(shape_cart.body, shape_cart)
        self.world.add(prismatic_joint_cart)
        self.world.add(shape_pole.body, shape_pole)
        self.world.add(revolute_joint_pole)
        self.world.add(shape_bob.body, shape_bob)
        self.world.add(pin_joint_bob, rotary_limit_joint_bob)  
        
        # Stack all shapes/joints in a list
        self.shapes = [shape_cart, shape_pole, shape_bob]
        self.joints = [prismatic_joint_cart, revolute_joint_pole, pin_joint_bob, rotary_limit_joint_bob]

        # Add collision labeling
        for shape in self.shapes:
            shape.pair_index = self.id 
            shape.collision_type = 1
        
    def destroy(self):
        # Remove this agent's physics objects after it fails.
        try:
            for joint in self.joints:
                self.world.remove(joint)
            for shape in self.shapes:
                self.world.remove(shape, shape.body)
        except:
            pass            

    def disturb(self, x, x_dot):
        # Occasionally push the bob if the cart is almost still, making the task harder.
        self.time_no_disturb -= 1
        self.time_no_disturb = np.clip(self.time_no_disturb, 0, Agent.time_no_disturb)
        if self.time_no_disturb == 0 and np.abs(x_dot) < 0.0125:
            sign_force = int(2 * (x < 0) - 1)
            F = sign_force * np.random.uniform(1.5 * self.max_force, 2.5 * self.max_force)
            self.time_no_disturb = Agent.time_no_disturb
            shape_bob = self.shapes[-1]
            shape_bob.body.apply_impulse_at_local_point((F, 0), (0, 0))
            shape_bob.color = (255, 255, 255, 255)

    def update(self):
        if not self.is_alive:
            return           

        # Read the physical state that will become the neural-network input.
        shape_cart = self.shapes[0]
        shape_pole = self.shapes[1]
        shape_bob = self.shapes[2]
        theta = shape_pole.body.angle
        theta_dot = shape_pole.body.angular_velocity
        x = shape_cart.body.position.x - self.position_of_agent[0]       
        x_dot = shape_cart.body.velocity.x    
        
        # The agent dies if the pole falls too far or the cart leaves the plateau.
        if np.abs(theta) > self.angle_treshold or np.abs(x) > (0.5 * Agent.width_plateau):
            self.is_alive = False
            self.destroy()
            return True
        
        # Normalize the four inputs before feeding them to the neural network.
        self.inputs[0] = theta  / self.angle_treshold
        self.inputs[1] = theta_dot
        self.inputs[2] = x / (0.5 * Agent.width_plateau)
        self.inputs[3] = x_dot / 200
		    
        # Convert the network output into a left/right impulse on the cart.
        outputs = self.neural_net.update(self.inputs)
        F = 2 * (outputs[0] - 0.5) * self.max_force
        shape_cart.body.apply_impulse_at_local_point((F, 0), (0, -0.5 * self.height_cart))
        # Set a constraint on the rotation of the cart
        shape_cart.body.angle = 0
        
        # Fitness is simply survival time: one point per update while alive.
        self.fitness += 1
        
        # Set state of color
        fraction = self.time_no_disturb / Agent.time_no_disturb
        shape_bob.color = (shape_bob.color[0], fraction * shape_bob.color[1], fraction * shape_bob.color[2], 255)  
        
        # Add, if appropriate, some disturbance    
        self.disturb(x, x_dot)
        
    def draw(self, interface, screen):
        Agent.draw_shapes(self.shapes, interface, screen)        
