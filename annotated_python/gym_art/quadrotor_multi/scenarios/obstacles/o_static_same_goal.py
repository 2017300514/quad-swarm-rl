import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base

# 这是障碍环境里的同目标静态基线。
# reset 时每架无人机都从自由栅格里拿到各自的出生点，但所有无人机共享同一个安全终点，
# 因此任务难点是“从分散起点穿过障碍汇聚到同一个目标区域”，而不是中途再切换任务。


class Scenario_o_static_same_goal(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        self.approch_goal_metric = 1.0

    def step(self):
        # 这个静态版本在 episode 中不额外改写 goals。
        # 外层环境只沿用 reset 时那一个共享终点推进障碍观测、碰撞和成功率统计。
        return

    def reset(self, obst_map=None, cell_centers=None):
        self.duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(self.duration_time * self.envs[0].control_freq)

        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None or cell_centers is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        # 出生点逐 agent 采样，但终点只采一个，并复制给全队。
        self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        self.end_point = self.max_square_area_center()

        # 这里保留统一场景接口需要的 formation 参数；真正生效的几何主体是 spawn/goal 点集。
        self.update_formation_and_relate_param()

        self.spawn_points = copy.deepcopy(self.start_point)
        self.goals = np.array([self.end_point for _ in range(self.num_agents)])
