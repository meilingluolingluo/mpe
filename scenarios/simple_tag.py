import numpy as np

from .._mpe_utils.rendering import Viewer

from .._mpe_utils.core import Agent, Landmark, World
from .._mpe_utils.scenario import BaseScenario

np_random = np.random.default_rng()  # Creating a new instance of a random generator

class Scenario(BaseScenario):
    def make_world(self, num_good=1, num_adversaries=3, num_obstacles=2, num_food=1):
        world = World()
        #viewer = Viewer(800, 600)  # 创建或获取一个 Viewer 实例
        # Set any world properties first
        world.dim_c = 2
        num_agents = num_adversaries + num_good

        # Add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = i < num_adversaries    
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if agent.adversary else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0  # 设置加速度
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # Add landmarks
        world.landmarks = [Landmark() for i in range(num_obstacles)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False  # 手动添加 boundary 属性

        # Add food
        world.food = [Landmark() for _ in range(num_food)]
        for i, food in enumerate(world.food):
            food.name = f'food_{i}'
            food.collide = False
            food.movable = False
            food.size = 0.03
            food.color = np.array([1.0, 1.0, 0.0])
            food.boundary = False  # 添加 boundary 属性

        # Combine landmarks and food into a single list
        world.landmarks.extend(world.food)

        return world

    def reset_world(self, world, np_random):
        # Random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # Set random initial states for agents, landmarks, and food
        for entity in world.landmarks + world.agents:
            entity.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            entity.state.p_vel = np.zeros(world.dim_p)
        # Ensure food color is set to yellow
        for food in world.food:
            food.color = np.array([1.0, 1.0, 0.0])  # 设置食物颜色为黄色

    def reset_food_position(self, food, world, np_random):
        food.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
        food.state.p_vel = np.zeros(world.dim_p)  # 假设食物不移动
        food.color = np.array([1.0, 1.0, 0.0])  # 保持食物颜色为黄色

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, entity1, entity2):
        delta_pos = entity1.state.p_pos - entity2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = entity1.size + entity2.size
        return dist < dist_min

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world, np_random):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world, np_random)
        )
        return main_reward

    def agent_reward(self, agent, world, np_random):
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if (
                shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                )
            # Penalty for colliding with obstacles
        for obstacle in world.landmarks:
            if obstacle.collide and self.is_collision(agent, obstacle):
                rew -= 10  # Subtract 5 from the reward for each collision with an obstacle

            # Penalty for being caught by adversaries
        for a in adversaries:
            if self.is_collision(a, agent):
                rew -= 10  # Subtract 10 from the reward for each collision with an adversary

        # 检测碰撞并重置食物位置
        for food in world.food:
            if self.is_collision(agent, food):
                rew += 50
                self.reset_food_position(food, world, np_random)

        # 计算最近食物距离的惩罚
        min_distance = min(
            np.sqrt(np.sum(np.square(food.state.p_pos - agent.state.p_pos)))
            for food in world.food
        )
        rew -= 0.05 * min_distance

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10) # 1 + (x - 1) * (x - 1)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if (
                shape
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min(
                    np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                    for a in agents
                )
        # Reward for catching good agents
        if agent.collide:
            for ag in agents:
                if self.is_collision(ag, agent):
                    rew += 5  # Reward for catching a good agent

        # Penalty for colliding with other adversaries
        for adv in adversaries:
            if adv != agent and self.is_collision(agent, adv):
                rew -= 2  # Subtract 2 for each collision with another adversary

        # Penalty for colliding with obstacles
        for obstacle in world.landmarks:
            if obstacle.collide and self.is_collision(agent, obstacle):
                rew -= 5  # Add a penalty for colliding with an obstacle
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        # 新增：添加食物位置到观察空间
        food_pos = [food.state.p_pos - agent.state.p_pos for food in world.food]
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)

        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + other_pos
            + other_vel
            + food_pos
        )
