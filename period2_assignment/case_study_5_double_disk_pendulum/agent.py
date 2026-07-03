import numpy as np
import pymunk
from neuralNetwork import Neural_network
import numpy as np
import pymunk
from neuralNetwork import Neural_network

class Agent():	
    counter         = -1  #gives each agent a unique ID
    is_initialized = False
    time_no_disturb = 500
    
    category_cart   = 0b0001
    mask_cart       = 0b1000
    category_disk   = 0b0010
    mask_disk       = 0b0000
    
    #convert pymunk coordinates to pygame coordinates
    @staticmethod
    def x2x_y2miny(v, interface):
        return v.x, interface.display.get_window_size()[1] - v.y

    #draws all objects to be displayed
    @staticmethod
    def draw_shapes(shapes, interface, screen):
        for shape in shapes: 
            v = shape.body.position

            #draws cart and plateu
            if type(shape) == pymunk.shapes.Poly: 
                V = [v_poly.rotated(shape.body.angle) + v for v_poly in shape.get_vertices()]              
                V = list(map(lambda v: Agent.x2x_y2miny(v, interface), V))                
                interface.draw.polygon(screen, shape.color, V, 0)
                interface.draw.aalines(screen, interface.Color('black'), True, V)
            
            #draws disks
            elif type(shape) == pymunk.shapes.Circle:
                rotation_vector = shape.body.rotation_vector   
                v_1 = Agent.x2x_y2miny(v, interface) 
                v_2 = Agent.x2x_y2miny(v + shape.radius * rotation_vector, interface) 
                interface.draw.circle(screen, shape.color, v_1, shape.radius)
                interface.draw.circle(screen, interface.Color('black'), v_1, shape.radius, 1)
                interface.draw.aaline(screen, interface.Color('black'), v_1, v_2)
    
    #prevents different agents from colliding with eachother
    @staticmethod
    def collision_callback(arbiter, world, data):
        collision_indices = []
        for shape in arbiter.shapes:
            collision_indices.append(shape.pair_index)
        return len(np.unique(collision_indices)) == 1    
    
    def __init__(self, position_of_agent, world):
        Agent.counter += 1
        
        if not Agent.is_initialized:
            Agent.width_plateau = 500
            Agent.height_plateau = 15
            Agent.body_plateau = pymunk.Body(body_type = pymunk.Body.STATIC)    
            Agent.body_plateau.position = (position_of_agent[0], position_of_agent[1] - 0.5 * Agent.height_plateau)
            Agent.shape_plateau = pymunk.Poly.create_box(Agent.body_plateau, (Agent.width_plateau, Agent.height_plateau))
            Agent.shape_plateau.friction = 0
            Agent.shape_plateau.color = (0, 255, 0, 255)
            world.add(Agent.body_plateau, Agent.shape_plateau)            
            world.on_collision(1, 1, begin = Agent.collision_callback)
            Agent.is_initialized = True

        #agent settings    
        self.id                = Agent.counter
        self.position_of_agent = position_of_agent     
        self.width_cart        = 75
        self.height_cart       = 30
        self.mass_cart         = 1
        self.radius_disk       = 30
        self.mass_disk         = 1
        self.max_force         = 50
        self.angle_treshold    = np.radians(15)

        # Neural network receives: 
        # disk1 angle, disk1 speed, disk2 angle, disk2 speed, cart position, cart speed
        self.neural_net        = Neural_network([6, 8, 1])
        self.inputs            = [None] * self.neural_net.number_of_inputs        
        self.world             = world
        self.reset()
    
    #reset agent for new attempt
    def reset(self):
        self.is_alive        = True
        self.fitness         = 0
        self.time_no_disturb = Agent.time_no_disturb     
        self.shapes          = []
        self.joints          = []
        self.create_agent()

    def create_agent(self):
        #creating cart
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

        #creating disk 1 (yellow disk)
        body_disk_1 = pymunk.Body()
        body_disk_1.position = (body_cart.position.x, body_cart.position.y + 0.5 * self.height_cart + self.radius_disk)
        shape_disk_1 = pymunk.Circle(body_disk_1, self.radius_disk)
        shape_disk_1.filter = pymunk.ShapeFilter(categories = Agent.category_disk, mask = Agent.mask_disk)
        shape_disk_1.mass = self.mass_disk
        shape_disk_1.color = (255, 255, 0, 255)
        joint_disk_1 = pymunk.PivotJoint(body_disk_1, body_cart, (0, -self.radius_disk), (0, 0.5 * self.height_cart))

        #creating disk 2 (red disk)
        body_disk_2 = pymunk.Body()
        body_disk_2.position = (body_disk_1.position.x, body_disk_1.position.y + 2 * self.radius_disk)
        shape_disk_2 = pymunk.Circle(body_disk_2, self.radius_disk)
        shape_disk_2.filter = pymunk.ShapeFilter(categories = Agent.category_disk, mask = Agent.mask_disk)
        shape_disk_2.mass = self.mass_disk
        shape_disk_2.color = (255, 0, 0, 255)
        joint_disk_2 = pymunk.PivotJoint(body_disk_2, body_disk_1, (0, -self.radius_disk), (0, self.radius_disk))
        
        #add objects to world
        self.world.add(body_cart, shape_cart, prismatic_joint_cart)
        self.world.add(body_disk_1, shape_disk_1, joint_disk_1)
        self.world.add(body_disk_2, shape_disk_2, joint_disk_2)
        
        #storing shapes and joints
        self.shapes = [shape_cart, shape_disk_1, shape_disk_2]
        self.joints = [prismatic_joint_cart, joint_disk_1, joint_disk_2]

        for shape in self.shapes:
            shape.pair_index = self.id 
            shape.collision_type = 1

    #removes the agent from the world after it dies   
    def destroy(self):
        try:
            for joint in self.joints:
                self.world.remove(joint)
            for shape in self.shapes:
                self.world.remove(shape, shape.body)
        except:
            pass            
    
    #applies disturbance after some time
    def disturb(self, x, x_dot):
        self.time_no_disturb -= 1
        self.time_no_disturb = np.clip(self.time_no_disturb, 0, Agent.time_no_disturb)

        if self.time_no_disturb == 0 and np.abs(x_dot) < 0.0125:
            sign_force = int(2 * (x < 0) - 1)

            # Small disturbance
            F = sign_force * np.random.uniform(0.1 * self.max_force, 0.3 * self.max_force)

            self.time_no_disturb = Agent.time_no_disturb

            # Apply small impulse to red disk
            self.shapes[2].body.apply_impulse_at_local_point((F, 0), (0, 0))
            self.shapes[2].color = (255, 255, 255, 255)

    def update(self):
        if not self.is_alive:
            return           

        shape_cart = self.shapes[0]
        disk_1 = self.shapes[1]
        disk_2 = self.shapes[2]

        theta_1 = disk_1.body.angle
        theta_1_dot = disk_1.body.angular_velocity
        theta_2 = disk_2.body.angle
        theta_2_dot = disk_2.body.angular_velocity
        x = shape_cart.body.position.x - self.position_of_agent[0]       
        x_dot = shape_cart.body.velocity.x    
        
        #death condition
        if (np.abs(theta_1) > self.angle_treshold 
            or np.abs(theta_2) > self.angle_treshold 
            or np.abs(x) > (0.5 * Agent.width_plateau)):

            self.is_alive = False
            self.destroy()
            return True
        
        self.inputs[0] = theta_1 / self.angle_treshold
        self.inputs[1] = theta_1_dot
        self.inputs[2] = theta_2 / self.angle_treshold
        self.inputs[3] = theta_2_dot
        self.inputs[4] = x / (0.5 * Agent.width_plateau)
        self.inputs[5] = x_dot / 200
		    
        outputs = self.neural_net.update(self.inputs)
        F = 2 * (outputs[0] - 0.5) * self.max_force
        shape_cart.body.apply_impulse_at_local_point((F, 0), (0, -0.5 * self.height_cart))
        shape_cart.body.angle = 0
        
        self.fitness += 1
        
        fraction = self.time_no_disturb / Agent.time_no_disturb
        disk_2.color = (disk_2.color[0], fraction * disk_2.color[1], fraction * disk_2.color[2], 255)  
        self.disturb(x, x_dot)
        
    def draw(self, interface, screen):
        Agent.draw_shapes(self.shapes, interface, screen)
