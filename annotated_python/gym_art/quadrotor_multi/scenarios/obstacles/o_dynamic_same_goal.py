import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base

# 这是障碍环境里的动态同目标场景。
# 它延续 `o_static_same_goal` 的“所有无人机共享一个目标”设定，但会周期性把这个共享目标重采到另一处自由空间，
# 相当于把普通 `dynamic_same_goal` 的任务节奏迁移到 obstacle map 约束下。


class Scenario_o_dynamic_same_goal(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        self.approch_goal_metric = 1.0

        # 新共享目标不能跳太远，否则在障碍环境里很容易直接变成不可跟踪任务。
        self.max_dist = 4.0

    def step(self):
        tick = self.envs[0].tick

        if tick % self.control_step_for_sec == 0 or tick == 1:
            new_goal = self.generate_pos_obst_map()
            while np.linalg.norm(self.end_point - new_goal) > self.max_dist:
                new_goal = self.generate_pos_obst_map()

            self.end_point = new_goal
            for i, env in enumerate(self.envs):
                env.goal = new_goal
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

        self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        self.end_point = self.max_square_area_center()

        self.update_formation_and_relate_param()

        self.spawn_points = copy.deepcopy(self.start_point)
        self.goals = np.array([self.end_point for _ in range(self.num_agents)])
