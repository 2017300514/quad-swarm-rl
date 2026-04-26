import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base

# 这是障碍环境里的目标交换场景。
# 起点仍从自由格中逐 agent 采样，但目标不是固定不变，而是在一组安全目标槽位内部周期性打乱，
# 因而它对应的是 obstacle map 约束下的 `swap_goals` 版本。


class Scenario_o_swap_goals(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        duration_time = 6.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_goals(self):
        # 这里不重采新的目标区域，只在已有安全槽位集合内部重新打乱映射。
        np.random.shuffle(self.goals)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self):
        tick = self.envs[0].tick
        if tick % self.control_step_for_sec == 0 and tick > 0:
            self.update_goals()

        return

    def reset(self, obst_map=None, cell_centers=None):
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        self.update_formation_and_relate_param()

        self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        self.spawn_points = copy.deepcopy(self.start_point)

        # 先在障碍图里找一个安全中心，再围绕它生成一组可交换目标槽位。
        self.formation_center = self.max_square_area_center()
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)
