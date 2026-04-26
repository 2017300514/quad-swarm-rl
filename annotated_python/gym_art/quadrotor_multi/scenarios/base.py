import numpy as np

from gym_art.quadrotor_multi.scenarios.utils import QUADS_PARAMS_DICT, update_formation_and_max_agent_per_layer, \
    update_layer_dist, get_formation_range, get_goal_by_formation
from gym_art.quadrotor_multi.scenarios.utils import generate_points, get_grid_dim_number

# 这个文件定义所有具体场景类共享的基类逻辑。
# 上游输入是 `quads_mode`、agent 数量和房间尺寸；下游产物是每架无人机的目标点排布，
# `quadrotor_multi.py` 会在 reset/step 时消费这些 goals 来设置任务目标和 episode 几何结构。


class QuadrotorScenario:
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        self.quads_mode = quads_mode
        self.envs = envs
        self.num_agents = num_agents
        self.room_dims = room_dims
        self.goals = None

        #  Set formation, num_agents_per_layer, lowest_formation_size, highest_formation_size, formation_size,
        #  layer_dist, formation_center
        #  Note: num_agents_per_layer for scalability, the maximum number of agent per layer
        self.formation = None
        self.formation_center = None
        self.lowest_formation_size, self.highest_formation_size = 1.0, 2.0
        self.formation_size = 1.0

        self.num_agents_per_layer = 8
        self.layer_dist = self.lowest_formation_size

        # Aux variables for scenario: pursuit evasion
        self.interp = None
        # Aux variables used in scenarios with obstacles
        self.spawn_points = None
        self.approch_goal_metric = 0.5

    def name(self):
        """
        :return: scenario name
        """
        return self.__class__.__name__

    def generate_goals(self, num_agents, formation_center=None, layer_dist=0.0):
        # 把“当前 episode 想采用的编队形状 + 尺寸”展开成每架无人机各自的 goal。
        # 这个函数是大多数场景 reset 的核心：不同编队最后都会在这里落到具体的三维目标坐标。
        if formation_center is None:
            formation_center = np.array([0., 0., 2.])

        if self.formation.startswith("circle"):
            # 对圆形编队，先决定每一层能放多少 agent，再按角度均匀铺开。
            if num_agents <= self.num_agents_per_layer:
                real_num_per_layer = [num_agents]
            else:
                whole_layer_num = num_agents // self.num_agents_per_layer
                real_num_per_layer = [self.num_agents_per_layer for _ in range(whole_layer_num)]
                rest_num = num_agents % self.num_agents_per_layer
                if rest_num > 0:
                    real_num_per_layer.append(rest_num)

            pi = np.pi
            goals = []
            for i in range(num_agents):
                cur_layer_num_agents = real_num_per_layer[i // self.num_agents_per_layer]
                degree = 2 * pi * (i % cur_layer_num_agents) / cur_layer_num_agents
                pos_0 = self.formation_size * np.cos(degree)
                pos_1 = self.formation_size * np.sin(degree)
                goal = get_goal_by_formation(formation=self.formation, pos_0=pos_0, pos_1=pos_1,
                                             layer_pos=(i // self.num_agents_per_layer) * layer_dist)
                goals.append(goal)

            goals = np.array(goals)
            goals += formation_center

        elif self.formation == "sphere":
            # 球面编队使用近似均匀点集，避免大量 agent 都堆在同一纬度附近。
            goals = self.formation_size * np.array(generate_points(num_agents)) + formation_center

        elif self.formation.startswith("grid"):
            # 网格编队会先把 agent 分层，再为每层求一个尽量接近方形的二维网格尺寸。
            if num_agents <= self.num_agents_per_layer:
                dim_1, dim_2 = get_grid_dim_number(num_agents)
                dim_size_each_layer = [[dim_1, dim_2]]
            else:
                # whole layer
                whole_layer_num = num_agents // self.num_agents_per_layer
                max_dim_1, max_dim_2 = get_grid_dim_number(self.num_agents_per_layer)
                dim_size_each_layer = [[max_dim_1, max_dim_2] for _ in range(whole_layer_num)]

                # deal with the rest of the drones
                rest_num = num_agents % self.num_agents_per_layer
                if rest_num > 0:
                    dim_1, dim_2 = get_grid_dim_number(rest_num)
                    dim_size_each_layer.append([dim_1, dim_2])

            goals = []
            for i in range(num_agents):
                dim_1, dim_2 = dim_size_each_layer[i // self.num_agents_per_layer]
                pos_0 = self.formation_size * (i % dim_2)
                pos_1 = self.formation_size * (int(i / dim_2) % dim_1)
                goal = get_goal_by_formation(formation=self.formation, pos_0=pos_0, pos_1=pos_1,
                                             layer_pos=(i // self.num_agents_per_layer) * layer_dist)
                goals.append(goal)

            mean_pos = np.mean(goals, axis=0)
            goals = goals - mean_pos + formation_center
        elif self.formation.startswith("cube"):
            # 立方体编队本质上是 3D 网格，同样在生成后回到 formation_center 周围。
            dim_size = np.power(num_agents, 1.0 / 3)
            floor_dim_size = int(dim_size)
            goals = []
            for i in range(num_agents):
                pos_0 = self.formation_size * (int(i / floor_dim_size) % floor_dim_size)
                pos_1 = self.formation_size * (i % floor_dim_size)
                goal = np.array(
                    [formation_center[2] + self.formation_size * (i // np.square(floor_dim_size)), pos_0, pos_1])
                goals.append(goal)

            mean_pos = np.mean(goals, axis=0)
            goals = goals - mean_pos + formation_center
        else:
            raise NotImplementedError("Unknown formation")

        return goals

    def update_formation_size(self, new_formation_size):
        # 动态场景可能在 episode 中间改变编队尺度。
        # 一旦尺度变化，就要立刻重算所有目标点，并同步写回每个子环境的 `env.goal`。
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                             layer_dist=self.layer_dist)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

    def update_formation_and_relate_param(self):
        # 这里把高层任务模式翻译成几何约束：
        # 先抽取本 episode 的编队类型，再结合 mode 允许的最近/最远间距范围采样 formation_size 和层间距。
        self.formation, self.num_agents_per_layer = update_formation_and_max_agent_per_layer(mode=self.quads_mode)
        lowest_dist, highest_dist = QUADS_PARAMS_DICT[self.quads_mode][1]
        self.lowest_formation_size, self.highest_formation_size = \
            get_formation_range(mode=self.quads_mode, formation=self.formation, num_agents=self.num_agents,
                                low=lowest_dist, high=highest_dist, num_agents_per_layer=self.num_agents_per_layer)

        self.formation_size = np.random.uniform(low=self.lowest_formation_size, high=self.highest_formation_size)
        self.layer_dist = update_layer_dist(low=self.lowest_formation_size, high=self.highest_formation_size)

    def step(self):
        raise NotImplementedError("Implemented in a specific scenario")

    def reset(self):
        # 默认 reset 会重新抽取编队类型和尺度，并把整个编队中心放回房间中部上方。
        self.update_formation_and_relate_param()

        self.formation_center = np.array([0.0, 0.0, 2.0])

        # 这里只生成并打乱 goal 集合，真正把第 i 个目标分配给第 i 架无人机的动作在多机环境 reset 中完成。
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)

    def standard_reset(self, formation_center=None):
        # 某些具体场景会传入自定义中心点，但其余流程与默认 reset 一样。
        self.update_formation_and_relate_param()

        if formation_center is None:
            self.formation_center = np.array([0.0, 0.0, 2.0])
        else:
            self.formation_center = formation_center

        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)
