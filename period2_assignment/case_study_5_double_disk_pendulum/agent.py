import numpy as np
import pymunk
from neuralNetwork import Neural_network


class Agent():
    counter = -1
    is_initialized = False
    time_no_disturb = 1500

    category_cart = 0b0001
    category_disk = 0b0010
    mask_none = 0b0000

    @staticmethod
    def x2x_y2miny(v, interface):
        return v.x, interface.display.get_window_size()[1] - v.y

    @staticmethod
    def draw_shapes(shapes, interface, screen):
        for shape in shapes:
            v = shape.body.position
            if type(shape) == pymunk.shapes.Poly:
                vertices = [v_poly.rotated(shape.body.angle) + v for v_poly in shape.get_vertices()]
                vertices = list(map(lambda p: Agent.x2x_y2miny(p, interface), vertices))
                interface.draw.polygon(screen, shape.color, vertices, 0)
                interface.draw.aalines(screen, interface.Color('black'), True, vertices)
            elif type(shape) == pymunk.shapes.Circle:
                center = Agent.x2x_y2miny(v, interface)
                radius = int(shape.radius)
                rotation_vector = shape.body.rotation_vector
                edge = Agent.x2x_y2miny(v + shape.radius * rotation_vector, interface)
                interface.draw.circle(screen, shape.color, center, radius)
                interface.draw.circle(screen, interface.Color('black'), center, radius, 2)
                interface.draw.aaline(screen, interface.Color('black'), center, edge)
            elif type(shape) == pymunk.shapes.Segment:
                rotation_vector = shape.body.rotation_vector
                p1 = Agent.x2x_y2miny(v + shape.a.cpvrotate(rotation_vector), interface)
                p2 = Agent.x2x_y2miny(v + shape.b.cpvrotate(rotation_vector), interface)
                interface.draw.line(screen, shape.color, p1, p2, int(2 * shape.radius))

    def __init__(self, position_of_agent, world):
        Agent.counter += 1
        self.id = Agent.counter
        self.position_of_agent = position_of_agent
        self.world = world

        if not Agent.is_initialized:
            Agent.width_plateau = 520
            Agent.height_plateau = 16
            Agent.body_plateau = pymunk.Body(body_type=pymunk.Body.STATIC)
            Agent.body_plateau.position = (position_of_agent[0], position_of_agent[1] - 0.5 * Agent.height_plateau)
            Agent.shape_plateau = pymunk.Poly.create_box(Agent.body_plateau, (Agent.width_plateau, Agent.height_plateau))
            Agent.shape_plateau.friction = 0
            Agent.shape_plateau.color = (0, 230, 40, 255)
            world.add(Agent.body_plateau, Agent.shape_plateau)
            Agent.is_initialized = True

        self.width_cart = 150
        self.height_cart = 42
        self.mass_cart = 2
        self.radius_disk = 36
        self.mass_disk = 1
        self.max_force = 90
        self.angle_threshold = np.radians(28)
        self.position_threshold = 0.5 * Agent.width_plateau - 35

        # Controller inputs:
        # lower angle, lower angular velocity, upper relative angle,
        # upper angular velocity, cart position, cart velocity.
        self.neural_net = Neural_network([6, 10, 1])
        self.activation_functions = [None, 'tanh', 'id']
        self.inputs = [0] * self.neural_net.number_of_inputs
        self.reset()

    def reset(self):
        self.is_alive = True
        self.fitness = 0
        self.time_no_disturb = Agent.time_no_disturb
        self.shapes = []
        self.joints = []
        self.create_agent()

    def body_position_from_anchor(self, world_anchor, local_anchor, angle):
        return pymunk.Vec2d(*world_anchor) - pymunk.Vec2d(*local_anchor).rotated(angle)

    def create_agent(self):
        body_cart = pymunk.Body()
        body_cart.position = (self.position_of_agent[0], self.position_of_agent[1] + 0.5 * self.height_cart)
        shape_cart = pymunk.Poly.create_box(body_cart, (self.width_cart, self.height_cart))
        shape_cart.filter = pymunk.ShapeFilter(categories=Agent.category_cart, mask=Agent.mask_none)
        shape_cart.mass = self.mass_cart
        shape_cart.friction = 0
        shape_cart.color = (55, 90, 255, 255)
        groove_joint_cart = pymunk.GrooveJoint(
            Agent.body_plateau,
            body_cart,
            (-0.5 * Agent.width_plateau, 0.5 * Agent.height_plateau),
            (0.5 * Agent.width_plateau, 0.5 * Agent.height_plateau),
            (0, -0.5 * self.height_cart),
        )

        cart_top = (body_cart.position.x, body_cart.position.y + 0.5 * self.height_cart)
        lower_angle = np.radians(4)
        upper_angle = np.radians(-3)

        body_lower = pymunk.Body()
        body_lower.angle = lower_angle
        body_lower.position = self.body_position_from_anchor(cart_top, (0, -self.radius_disk), lower_angle)
        shape_lower = pymunk.Circle(body_lower, self.radius_disk)
        shape_lower.filter = pymunk.ShapeFilter(categories=Agent.category_disk, mask=Agent.mask_none)
        shape_lower.mass = self.mass_disk
        shape_lower.color = (255, 240, 0, 255)
        joint_lower = pymunk.PivotJoint(body_lower, body_cart, (0, -self.radius_disk), (0, 0.5 * self.height_cart))

        lower_top_world = body_lower.local_to_world((0, self.radius_disk))
        body_upper = pymunk.Body()
        body_upper.angle = upper_angle
        body_upper.position = self.body_position_from_anchor(lower_top_world, (0, -self.radius_disk), upper_angle)
        shape_upper = pymunk.Circle(body_upper, self.radius_disk)
        shape_upper.filter = pymunk.ShapeFilter(categories=Agent.category_disk, mask=Agent.mask_none)
        shape_upper.mass = self.mass_disk
        shape_upper.color = (255, 40, 30, 255)
        joint_upper = pymunk.PivotJoint(body_upper, body_lower, (0, -self.radius_disk), (0, self.radius_disk))

        rod_lower = pymunk.Segment(body_lower, (0, -self.radius_disk), (0, self.radius_disk), 2)
        rod_lower.filter = pymunk.ShapeFilter(categories=Agent.category_disk, mask=Agent.mask_none)
        rod_lower.color = (30, 30, 30, 255)
        rod_upper = pymunk.Segment(body_upper, (0, -self.radius_disk), (0, self.radius_disk), 2)
        rod_upper.filter = pymunk.ShapeFilter(categories=Agent.category_disk, mask=Agent.mask_none)
        rod_upper.color = (30, 30, 30, 255)

        self.world.add(body_cart, shape_cart, groove_joint_cart)
        self.world.add(body_lower, shape_lower, rod_lower, joint_lower)
        self.world.add(body_upper, shape_upper, rod_upper, joint_upper)

        self.shapes = [shape_cart, rod_lower, shape_lower, rod_upper, shape_upper]
        self.joints = [groove_joint_cart, joint_lower, joint_upper]

    def destroy(self):
        try:
            for joint in self.joints:
                self.world.remove(joint)
            bodies = []
            for shape in self.shapes:
                if shape.body not in bodies:
                    bodies.append(shape.body)
                self.world.remove(shape)
            for body in bodies:
                self.world.remove(body)
        except Exception:
            pass

    def disturb(self, cart_x, cart_x_dot):
        # A disturbance impulse is applied to the upper disk after the controller
        # has had time to settle. This tests the robustness of the evolved ENN.
        self.time_no_disturb -= 1
        self.time_no_disturb = np.clip(self.time_no_disturb, 0, Agent.time_no_disturb)
        if self.time_no_disturb == 0 and abs(cart_x_dot) < 0.15:
            sign_force = int(2 * (cart_x < 0) - 1)
            impulse = sign_force * np.random.uniform(1.4 * self.max_force, 2.2 * self.max_force)
            upper_disk = self.shapes[-1]
            upper_disk.body.apply_impulse_at_local_point((impulse, 0), (0, 0))
            upper_disk.color = (255, 255, 255, 255)
            self.time_no_disturb = Agent.time_no_disturb

    def update(self):
        if not self.is_alive:
            return False

        shape_cart = self.shapes[0]
        shape_lower = self.shapes[2]
        shape_upper = self.shapes[4]

        lower_angle = shape_lower.body.angle
        lower_speed = shape_lower.body.angular_velocity
        relative_upper_angle = shape_upper.body.angle - shape_lower.body.angle
        upper_speed = shape_upper.body.angular_velocity
        cart_x = shape_cart.body.position.x - self.position_of_agent[0]
        cart_x_dot = shape_cart.body.velocity.x

        if (
            abs(lower_angle) > self.angle_threshold
            or abs(relative_upper_angle) > self.angle_threshold
            or abs(cart_x) > self.position_threshold
        ):
            self.is_alive = False
            self.destroy()
            return True

        self.inputs[0] = lower_angle / self.angle_threshold
        self.inputs[1] = lower_speed / 8.0
        self.inputs[2] = relative_upper_angle / self.angle_threshold
        self.inputs[3] = upper_speed / 8.0
        self.inputs[4] = cart_x / self.position_threshold
        self.inputs[5] = cart_x_dot / 260.0

        output = self.neural_net.update(self.inputs, self.activation_functions)[0]
        force = np.tanh(output) * self.max_force
        shape_cart.body.apply_impulse_at_local_point((force, 0), (0, 0))
        shape_cart.body.angle = 0

        balance_penalty = 0.25 * abs(lower_angle) + 0.25 * abs(relative_upper_angle) + 0.001 * abs(cart_x)
        self.fitness += max(0.0, 1.0 - balance_penalty)

        fraction = self.time_no_disturb / Agent.time_no_disturb
        shape_upper.color = (255, int(40 + 215 * (1 - fraction)), int(30 + 225 * (1 - fraction)), 255)
        self.disturb(cart_x, cart_x_dot)
        return False

    def draw(self, interface, screen):
        Agent.draw_shapes(self.shapes, interface, screen)
