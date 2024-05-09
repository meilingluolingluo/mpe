import numpy as np


class EntityState:  # physical/external base state of all entities
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        #self.p_vel = None
        self.p_vel = np.zeros(2)  # 假设是二维空间
        # Add orientation angle
        self.angle = 0  # Default angle


class AgentState(
    EntityState
):  # state of agents (including communication and internal/mental state)
    def __init__(self):
        super().__init__()
        # communication utterance
        self.c = None
        self.target_angle = 0  # 添加目标角度属性
        self.max_angle_speed = np.radians(3)  # 最大角速度，比如每秒30度

class Action:  # action of the agent
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


class Entity:  # properties and state of physical world entity
    def __init__(self):
        # name
        self.name = ""
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


class Landmark(Entity):  # properties of landmark entities
    def __init__(self):
        super().__init__()


class Agent(Entity):  # properties of agent entities
    def __init__(self):
        super().__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None


class World:  # multi-agent world
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    # def apply_action_force(self, p_force):
    #     # set applied forces
    #     for i, agent in enumerate(self.agents):
    #         if agent.movable:
    #             noise = (
    #                 np.random.randn(*agent.action.u.shape) * agent.u_noise
    #                 if agent.u_noise
    #                 else 0.0
    #             )
    #             p_force[i] = agent.action.u + noise
    #     return p_force
    def apply_action_force(self, p_force):
        # 设置应用的力
        for i, agent in enumerate(self.agents):
            if agent.movable:
                # 如果是第一次设置速度，则直接使用动作中的速度向量
                if agent.state.p_vel is None:
                    agent.state.p_vel = agent.action.u
                else:
                    # 应用平滑因子来平滑速度变化
                    smoothing_factor = 0.8  # 可以调整这个值以获得不同程度的平滑效果
                    # 确保action.u已经正确赋值
                    if agent.action.u is not None:
                        # 平滑处理，将当前速度和新速度合并
                        agent.state.p_vel = agent.state.p_vel * smoothing_factor + agent.action.u * (
                                    1 - smoothing_factor)

                # 添加物理动力噪声（如果有的话）
                noise = np.random.randn(*agent.state.p_vel.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.state.p_vel + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.linalg.norm(entity.state.p_vel)
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / speed * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

            # Smooth angle update based on target angle with speed limit
            if np.linalg.norm(entity.state.p_vel) > 0:
                desired_angle = np.arctan2(entity.state.p_vel[1], entity.state.p_vel[0])
                current_angle = np.radians(entity.state.angle)
                angle_diff = (desired_angle - current_angle + np.pi) % (2 * np.pi) - np.pi
                max_angle_change = entity.state.max_angle_speed * self.dt
                angle_change = np.clip(angle_diff, -max_angle_change, max_angle_change)
                new_angle = (current_angle + angle_change) % (2 * np.pi)
                entity.state.angle = np.degrees(new_angle)



    # integrate physical state
    # def integrate_state(self, p_force):
    #     for i, entity in enumerate(self.entities):
    #         if not entity.movable:
    #             continue
    #         entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
    #         if p_force[i] is not None:
    #             entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
    #         if entity.max_speed is not None:
    #             speed = np.sqrt(
    #                 np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
    #             )
    #
    #             # if speed > entity.max_speed:
    #             #     entity.state.p_vel = (
    #             #         entity.state.p_vel
    #             #         / np.sqrt(
    #             #             np.square(entity.state.p_vel[0])
    #             #             + np.square(entity.state.p_vel[1])
    #             #         )
    #             #         * entity.max_speed
    #             #     )
    #             if speed > entity.max_speed:
    #                 entity.state.p_vel = (
    #                         entity.state.p_vel / speed * entity.max_speed
    #                 )
    #         entity.state.p_pos += entity.state.p_vel * self.dt
    #         # Update angle based on velocity direction
    #         # if np.linalg.norm(entity.state.p_vel) > 0:
    #         #     entity.state.angle = np.arctan2(entity.state.p_vel[1], entity.state.p_vel[0])
    #         # Smooth angle update based on target angle
    #         if np.linalg.norm(entity.state.p_vel) > 0:
    #             desired_angle = np.arctan2(entity.state.p_vel[1], entity.state.p_vel[0])
    #             current_angle = entity.state.angle
    #             # Compute the shortest angular distance
    #             angle_diff = (desired_angle - current_angle + np.pi) % (2 * np.pi) - np.pi
    #             entity.state.angle += angle_diff * 0.1  # Adjust the angle by a small step
    #             entity.state.angle %= 2 * np.pi  # Normalize angle

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = (
                np.random.randn(*agent.action.c.shape) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise



    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
