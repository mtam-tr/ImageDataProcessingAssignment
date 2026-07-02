import numpy as np
import pymunk
from neuralNetwork import Neural_network


class Agent():
    """Double-disk version of the provided inverted-pendulum ENN agent.

    The original example balances one pole/bob on a cart. This case keeps the
    same PyMunk cart, ENN controller, genetic fitness, disturbance, and drawing
    structure, but replaces the pole/bob with two hinged disks.
    """
    caption = 'Case 5 - Double Disk Pendulum ENN'
    window_size = (900, 700)
    frames_per_second = 60
    population_size = 40
    physics_steps_per_second = 120
    generation_steps = 1800
    start_position = (window_size[0] // 2, 170)
    width_plateau = 520
    height_plateau = 16

    counter = -1
    is_initialized = False
    time_no_disturb = 1500

    category_cart = 0b0001
    category_disk = 0b0010
    mask_none = 0b0000

    # Convenience function from the example: convert PyMunk's positive-y-up
    # coordinates to Pygame's positive-y-down screen coordinates.
    @staticmethod
    def x2x_y2miny(v, interface):
        return v.x, interface.display.get_window_size()[1] - v.y

    # Draw PyMunk shapes with Pygame, extended from the example to show disk
    # rotation using a black radial line.
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

    # Keep the collision-labeling hook from the example. The case 5 shapes are
    # filtered not to collide, but labeling still documents the shared scaffold.
    @staticmethod
    def collision_callback(arbiter, world, data):
        collision_indices = []
        for shape in arbiter.shapes:
            collision_indices.append(shape.pair_index)
        return len(np.unique(collision_indices)) == 1

    def __init__(self, position_of_agent=None, world=None):
        Agent.counter += 1
        self.id = Agent.counter
        if position_of_agent is None:
            position_of_agent = Agent.start_position

        self.position_of_agent = position_of_agent
        self.owns_world = world is None

        if world is None:
            world = pymunk.Space()
            world.gravity = (0, -981)
            world.damping = 0.995
            world.iterations = 16

        self.world = world

        self.create_plateau()

        self.width_cart = 150
        self.height_cart = 42
        self.mass_cart = 2
        self.radius_disk = 36
        self.mass_disk = 1
        self.max_force = 90
        self.angle_threshold = np.radians(28)
        self.position_threshold = 0.5 * Agent.width_plateau - 35

        # Controller network, expanded from the example's [4, 8, 1].
        # Controller inputs:
        # lower angle, lower angular velocity, upper relative angle,
        # upper angular velocity, cart position, cart velocity.
        self.neural_net = Neural_network([6, 10, 1])
        self.inputs = [0] * self.neural_net.number_of_inputs
        self.snapshot = None
        self.error = float('inf')
        self.reset()

    def create_plateau(self):
        if self.owns_world:
            self.body_plateau = pymunk.Body(body_type=pymunk.Body.STATIC)
            self.body_plateau.position = (
                self.position_of_agent[0],
                self.position_of_agent[1] - 0.5 * Agent.height_plateau
            )
            self.shape_plateau = pymunk.Poly.create_box(
                self.body_plateau,
                (Agent.width_plateau, Agent.height_plateau)
            )
            self.shape_plateau.friction = 0
            self.shape_plateau.color = (0, 230, 40, 255)
            self.world.add(self.body_plateau, self.shape_plateau)
            return

        if not Agent.is_initialized:
            Agent.body_plateau = pymunk.Body(body_type=pymunk.Body.STATIC)
            Agent.body_plateau.position = (
                self.position_of_agent[0],
                self.position_of_agent[1] - 0.5 * Agent.height_plateau
            )
            Agent.shape_plateau = pymunk.Poly.create_box(
                Agent.body_plateau,
                (Agent.width_plateau, Agent.height_plateau)
            )
            Agent.shape_plateau.friction = 0
            Agent.shape_plateau.color = (0, 230, 40, 255)
            self.world.add(Agent.body_plateau, Agent.shape_plateau)
            Agent.is_initialized = True

        self.body_plateau = Agent.body_plateau
        self.shape_plateau = Agent.shape_plateau

    def reset(self):
        if hasattr(self, 'shapes'):
            self.destroy()

        self.is_alive = True
        self.fitness = 0
        self.error = float('inf')
        self.time_no_disturb = Agent.time_no_disturb
        self.shapes = []
        self.joints = []
        self.create_agent()

    def body_position_from_anchor(self, world_anchor, local_anchor, angle):
        return pymunk.Vec2d(*world_anchor) - pymunk.Vec2d(*local_anchor).rotated(angle)

    def create_agent(self):
        # Cart: same horizontal moving base as the example, constrained to slide
        # along the plateau by a groove joint.
        body_cart = pymunk.Body()
        body_cart.position = (self.position_of_agent[0], self.position_of_agent[1] + 0.5 * self.height_cart)
        shape_cart = pymunk.Poly.create_box(body_cart, (self.width_cart, self.height_cart))
        shape_cart.filter = pymunk.ShapeFilter(categories=Agent.category_cart, mask=Agent.mask_none)
        shape_cart.mass = self.mass_cart
        shape_cart.friction = 0
        shape_cart.color = (55, 90, 255, 255)
        groove_joint_cart = pymunk.GrooveJoint(
            self.body_plateau,
            body_cart,
            (-0.5 * Agent.width_plateau, 0.5 * Agent.height_plateau),
            (0.5 * Agent.width_plateau, 0.5 * Agent.height_plateau),
            (0, -0.5 * self.height_cart),
        )

        cart_top = (body_cart.position.x, body_cart.position.y + 0.5 * self.height_cart)
        lower_angle = np.radians(4)
        upper_angle = np.radians(-3)

        # Lower disk: replaces the example's pole. It is hinged to the cart and
        # starts slightly off vertical so the controller must react.
        body_lower = pymunk.Body()
        body_lower.angle = lower_angle
        body_lower.position = self.body_position_from_anchor(cart_top, (0, -self.radius_disk), lower_angle)
        shape_lower = pymunk.Circle(body_lower, self.radius_disk)
        shape_lower.filter = pymunk.ShapeFilter(categories=Agent.category_disk, mask=Agent.mask_none)
        shape_lower.mass = self.mass_disk
        shape_lower.color = (255, 240, 0, 255)
        joint_lower = pymunk.PivotJoint(body_lower, body_cart, (0, -self.radius_disk), (0, 0.5 * self.height_cart))

        lower_top_world = body_lower.local_to_world((0, self.radius_disk))
        # Upper disk: replaces the example's bob, but remains free to rotate as
        # a second inverted-pendulum link instead of being locked to a pole.
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

        # Same collision labeling pattern as the example. It keeps each agent's
        # shapes identifiable when several candidates share one world.
        for shape in self.shapes:
            shape.pair_index = self.id
            shape.collision_type = 1

        self.capture_snapshot()

    def destroy(self):
        bodies = []

        for joint in list(getattr(self, 'joints', [])):
            try:
                self.world.remove(joint)
            except Exception:
                pass

        for shape in list(getattr(self, 'shapes', [])):
            try:
                if shape.body not in bodies:
                    bodies.append(shape.body)
                self.world.remove(shape)
            except Exception:
                pass

        for body in bodies:
            try:
                self.world.remove(body)
            except Exception:
                pass

        self.shapes = []
        self.joints = []

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
            upper_disk.color = (255, 40, 30, 255)
            self.time_no_disturb = Agent.time_no_disturb

    def update(self):
        # Existing case-study main calls Agent() without a shared world, so one
        # full episode is evaluated off-screen. If a shared world is supplied,
        # this behaves like the original example and advances one live step.
        if self.owns_world:
            self.evaluate_episode()
            return not self.is_alive

        return self.update_step()

    def evaluate_episode(self):
        self.reset()

        for _ in range(Agent.generation_steps):
            if self.update_step():
                break
            self.world.step(1 / Agent.physics_steps_per_second)

        if self.is_alive:
            self.capture_snapshot()

        self.error = 1.0 - min(1.0, self.fitness / Agent.generation_steps)

    def update_step(self):
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

        # The agent dies if either disk falls too far or the cart leaves the
        # allowed platform range.
        if (
            abs(lower_angle) > self.angle_threshold
            or abs(relative_upper_angle) > self.angle_threshold
            or abs(cart_x) > self.position_threshold
        ):
            self.is_alive = False
            self.capture_snapshot()
            self.destroy()
            return True

        # Normalize the physical state before feeding it to the neural network.
        self.inputs[0] = lower_angle / self.angle_threshold
        self.inputs[1] = lower_speed / 8.0
        self.inputs[2] = relative_upper_angle / self.angle_threshold
        self.inputs[3] = upper_speed / 8.0
        self.inputs[4] = cart_x / self.position_threshold
        self.inputs[5] = cart_x_dot / 260.0

        # Convert the network output into the left/right cart impulse, following
        # the example's force-control idea.
        output = self.neural_net.update(self.inputs)[0]
        force = (2.0 * output - 1.0) * self.max_force
        shape_cart.body.apply_impulse_at_local_point((force, 0), (0, 0))
        shape_cart.body.angle = 0

        balance_penalty = 0.25 * abs(lower_angle) + 0.25 * abs(relative_upper_angle) + 0.001 * abs(cart_x)
        self.fitness += max(0.0, 1.0 - balance_penalty)

        shape_upper.color = (255, 40, 30, 255)
        self.disturb(cart_x, cart_x_dot)
        return False

    def capture_snapshot(self):
        if not self.shapes:
            return

        shape_cart = self.shapes[0]
        shape_lower = self.shapes[2]
        shape_upper = self.shapes[4]

        self.snapshot = {
            'cart_position': (shape_cart.body.position.x, shape_cart.body.position.y),
            'cart_angle': shape_cart.body.angle,
            'lower_position': (shape_lower.body.position.x, shape_lower.body.position.y),
            'lower_angle': shape_lower.body.angle,
            'upper_position': (shape_upper.body.position.x, shape_upper.body.position.y),
            'upper_angle': shape_upper.body.angle,
            'upper_color': tuple(shape_upper.color),
        }

    def __deepcopy__(self, memo):
        copied_agent = Agent.__new__(Agent)
        copied_agent.fitness = self.fitness
        copied_agent.error = self.error
        copied_agent.snapshot = self.snapshot.copy() if self.snapshot is not None else None
        copied_agent.shapes = []
        copied_agent.joints = []
        copied_agent.shape_plateau = self.shape_plateau
        copied_agent.width_cart = self.width_cart
        copied_agent.height_cart = self.height_cart
        copied_agent.radius_disk = self.radius_disk
        return copied_agent

    def draw_snapshot(self, interface, screen):
        if self.snapshot is None:
            return

        Agent.draw_shapes([self.shape_plateau], interface, screen)

        cart_position = pymunk.Vec2d(*self.snapshot['cart_position'])
        lower_position = pymunk.Vec2d(*self.snapshot['lower_position'])
        upper_position = pymunk.Vec2d(*self.snapshot['upper_position'])

        cart_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        cart_body.position = cart_position
        cart_body.angle = self.snapshot['cart_angle']
        cart_shape = pymunk.Poly.create_box(cart_body, (self.width_cart, self.height_cart))
        cart_shape.color = (55, 90, 255, 255)

        lower_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        lower_body.position = lower_position
        lower_body.angle = self.snapshot['lower_angle']
        lower_shape = pymunk.Circle(lower_body, self.radius_disk)
        lower_shape.color = (255, 240, 0, 255)
        lower_rod = pymunk.Segment(lower_body, (0, -self.radius_disk), (0, self.radius_disk), 2)
        lower_rod.color = (30, 30, 30, 255)

        upper_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        upper_body.position = upper_position
        upper_body.angle = self.snapshot['upper_angle']
        upper_shape = pymunk.Circle(upper_body, self.radius_disk)
        upper_shape.color = self.snapshot['upper_color']
        upper_rod = pymunk.Segment(upper_body, (0, -self.radius_disk), (0, self.radius_disk), 2)
        upper_rod.color = (30, 30, 30, 255)

        Agent.draw_shapes(
            [cart_shape, lower_rod, lower_shape, upper_rod, upper_shape],
            interface,
            screen
        )

    def draw(self, interface, screen, generation=None):
        if not self.shapes:
            self.draw_snapshot(interface, screen)
            return

        Agent.draw_shapes([self.shape_plateau], interface, screen)
        Agent.draw_shapes(self.shapes, interface, screen)
