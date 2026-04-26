import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario
from gym_art.quadrotor_multi.scenarios.utils import QUADS_MODE_LIST_SINGLE, QUADS_MODE_LIST, \
    QUADS_MODE_LIST_OBSTACLES, QUADS_MODE_LIST_OBSTACLES_SINGLE

# Neighbor Scenarios
from gym_art.quadrotor_multi.scenarios.static_same_goal import Scenario_static_same_goal
from gym_art.quadrotor_multi.scenarios.dynamic_diff_goal import Scenario_dynamic_diff_goal
from gym_art.quadrotor_multi.scenarios.dynamic_formations import Scenario_dynamic_formations
from gym_art.quadrotor_multi.scenarios.dynamic_same_goal import Scenario_dynamic_same_goal
from gym_art.quadrotor_multi.scenarios.ep_lissajous3D import Scenario_ep_lissajous3D
from gym_art.quadrotor_multi.scenarios.ep_rand_bezier import Scenario_ep_rand_bezier
from gym_art.quadrotor_multi.scenarios.run_away import Scenario_run_away
from gym_art.quadrotor_multi.scenarios.static_diff_goal import Scenario_static_diff_goal
from gym_art.quadrotor_multi.scenarios.static_same_goal import Scenario_static_same_goal
from gym_art.quadrotor_multi.scenarios.swap_goals import Scenario_swap_goals
from gym_art.quadrotor_multi.scenarios.swarm_vs_swarm import Scenario_swarm_vs_swarm

# Obstacles
from gym_art.quadrotor_multi.scenarios.obstacles.o_random import Scenario_o_random
from gym_art.quadrotor_multi.scenarios.obstacles.o_static_same_goal import Scenario_o_static_same_goal
from gym_art.quadrotor_multi.scenarios.obstacles.o_dynamic_same_goal import Scenario_o_dynamic_same_goal
from gym_art.quadrotor_multi.scenarios.obstacles.o_swap_goals import Scenario_o_swap_goals
from gym_art.quadrotor_multi.scenarios.obstacles.o_ep_rand_bezier import Scenario_o_ep_rand_bezier

# Test Scenarios
from gym_art.quadrotor_multi.scenarios.test.o_test import Scenario_o_test

# 这个文件是场景分发器。
# 上游只给一个抽象模式 `mix` 和当前环境配置；这里负责在每个 episode 开始时实际挑选某个具体 scenario，
# 再把它生成出的 goals、spawn_points 和 formation_size 回传给多机环境主循环。


def create_scenario(quads_mode, envs, num_agents, room_dims):
    # 项目里约定具体场景类名都叫 `Scenario_<mode>`，这里用这个命名约定动态实例化真实场景。
    cls = eval('Scenario_' + quads_mode)
    scenario = cls(quads_mode, envs, num_agents, room_dims)
    return scenario


class Scenario_mix(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)

        # 根据“单机/多机”和“是否有障碍物”先决定本实验允许抽到哪些场景。
        # 这样 `mix` 可以作为一个上层入口，让训练在多个任务族之间自动切换。
        if num_agents == 1:
            if envs[0].use_obstacles:
                self.quads_mode_list = QUADS_MODE_LIST_OBSTACLES_SINGLE
            else:
                self.quads_mode_list = QUADS_MODE_LIST_SINGLE
        elif num_agents > 1 and not envs[0].use_obstacles:
            self.quads_mode_list = QUADS_MODE_LIST
        elif envs[0].use_obstacles:
            self.quads_mode_list = QUADS_MODE_LIST_OBSTACLES
        else:
            raise NotImplementedError("Unknown!")

        self.scenario = None
        self.approch_goal_metric = 0.5

        self.spawn_points = None

    def name(self):
        """
        :return: the name of the actual scenario used in this episode
        """
        return self.scenario.__class__.__name__

    def step(self):
        # `mix` 本身不定义场景动力学，真正的时序逻辑委托给当前被选中的具体 scenario。
        self.scenario.step()

        # 某些动态场景会在 step 中修改 goals，这里把变化同步回最外层环境。
        self.goals = self.scenario.goals

        # formation_size 主要给可视化和上层统计使用。
        self.formation_size = self.scenario.formation_size
        return

    def reset(self, obst_map=None, cell_centers=None):
        # 每个 episode 随机选一个真实任务模式，相当于在训练集里做任务级 domain randomization。
        mode_index = np.random.randint(low=0, high=len(self.quads_mode_list))
        mode = self.quads_mode_list[mode_index]

        self.scenario = create_scenario(quads_mode=mode, envs=self.envs, num_agents=self.num_agents,
                                        room_dims=self.room_dims)

        # 有障碍物的场景 reset 需要额外拿到障碍物地图与网格中心。
        if obst_map is not None:
            self.scenario.reset(obst_map, cell_centers)
        else:
            self.scenario.reset()

        # 这些字段是多机环境 reset 后会立即消费的场景输出。
        self.goals = self.scenario.goals
        self.spawn_points = self.scenario.spawn_points
        self.formation_size = self.scenario.formation_size
        self.approch_goal_metric = self.scenario.approch_goal_metric
