# 中文注释副本；原始文件：gym_art/quadrotor_multi/scenarios/obstacles/o_ep_rand_bezier.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np
import copy
import bezier

# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


# 定义类 `Scenario_o_ep_rand_bezier`。
class Scenario_o_ep_rand_bezier(Scenario_o_base):
    # 定义函数 `__init__`。
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        # 调用 `super` 执行当前处理。
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # teleport every [4.0, 6.0] secs
        # 保存或更新 `duration_time` 的值。
        duration_time = 0.3
        # 保存或更新 `control_step_for_sec` 的值。
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        # 保存或更新 `approch_goal_metric` 的值。
        self.approch_goal_metric = 1.0

    # 定义函数 `step`。
    def step(self):
        # randomly sample new goal pos in free space and have the goal move there following a bezier curve
        # 保存或更新 `tick` 的值。
        tick = self.envs[0].tick
        # 保存或更新 `control_freq` 的值。
        control_freq = self.envs[0].control_freq
        # 保存或更新 `num_secs` 的值。
        num_secs = 6
        # 保存或更新 `control_steps` 的值。
        control_steps = int(num_secs * control_freq)
        # 保存或更新 `t` 的值。
        t = tick % control_steps
        # 保存或更新 `room_dims` 的值。
        room_dims = np.array(self.room_dims) - self.formation_size
        # min and max distance the goal can spawn away from its current location. 30 = empirical upper bound on
        # velocity that the drones can handle.
        # 保存或更新 `max_dist` 的值。
        max_dist = min(5, max(room_dims))
        # 保存或更新 `min_dist` 的值。
        min_dist = max_dist / 2
        # 根据条件决定是否进入当前分支。
        if tick % control_steps == 0 or tick == 1:
            # sample a new goal pos that's within the room boundaries and satisfies the distance constraint
            # 保存或更新 `new_goal_found` 的值。
            new_goal_found = False
            # 在条件成立时持续执行下面的循环体。
            while not new_goal_found:
                # 同时更新 `low`, `high` 等变量。
                low, high = np.array([-room_dims[0] / 2, -room_dims[1] / 2, 1.5]), np.array(
                    [room_dims[0] / 2, room_dims[1] / 2, 3.0])
                # need an intermediate point for a deg=2 curve
                # 保存或更新 `new_pos` 的值。
                new_pos = np.random.uniform(low=-high, high=high, size=(2, 3)).reshape(3, 2)
                # add some velocity randomization = random magnitude * unit direction
                # 保存或更新 `new_pos` 的值。
                new_pos = new_pos * np.random.randint(min_dist, max_dist + 1) / np.linalg.norm(new_pos, axis=0)
                # 保存或更新 `new_pos` 的值。
                new_pos = self.goals[0].reshape(3, 1) + new_pos
                # 保存或更新 `lower_bound` 的值。
                lower_bound = np.expand_dims(low, axis=1)
                # 保存或更新 `upper_bound` 的值。
                upper_bound = np.expand_dims(high, axis=1)
                # 保存或更新 `new_goal_found` 的值。
                new_goal_found = (new_pos > lower_bound + 0.5).all() and (
                            new_pos < upper_bound - 0.5).all()  # check bounds that are slightly smaller than the room dims
            # new_pos = np.append(self.sampled_points[t], 2.0)
            # new_pos = new_pos.reshape(3, 1)
            # 保存或更新 `nodes` 的值。
            nodes = np.concatenate((self.goals[0].reshape(3, 1), new_pos), axis=1)
            # 保存或更新 `nodes` 的值。
            nodes = np.asfortranarray(nodes)
            # 保存或更新 `pts` 的值。
            pts = np.linspace(0, 1, control_steps)
            # 保存或更新 `curve` 的值。
            curve = bezier.Curve(nodes, degree=2)
            # 保存或更新 `interp` 的值。
            self.interp = curve.evaluate_multi(pts)
            # self.interp = np.clip(self.interp, a_min=np.array([0,0,0.2]).reshape(3,1), a_max=high.reshape(3,
            # 1)) # want goal clipping to be slightly above the floor
        # 根据条件决定是否进入当前分支。
        if tick % control_steps != 0 and tick > 1:
            # 保存或更新 `goals` 的值。
            self.goals = np.array([self.interp[:, t] for _ in range(self.num_agents)])

            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i, env in enumerate(self.envs):
                # 保存或更新 `env.goal` 的值。
                env.goal = self.goals[i]

        # 返回当前函数的结果。
        return

    # 定义函数 `reset`。
    def reset(self, obst_map=None, cell_centers=None):
        # Update duration time
        # 保存或更新 `duration_time` 的值。
        self.duration_time = 0.01
        # 保存或更新 `control_step_for_sec` 的值。
        self.control_step_for_sec = int(self.duration_time * self.envs[0].control_freq)

        # 保存或更新 `obstacle_map` 的值。
        self.obstacle_map = obst_map
        # 保存或更新 `cell_centers` 的值。
        self.cell_centers = cell_centers
        # 根据条件决定是否进入当前分支。
        if obst_map is None or cell_centers is None:
            # 主动抛出异常以中止或提示错误。
            raise NotImplementedError

        # 执行这一行逻辑。
        obst_map_locs = np.where(self.obstacle_map == 0)
        # 保存或更新 `free_space` 的值。
        self.free_space = list(zip(*obst_map_locs))

        # 保存或更新 `start_point` 的值。
        self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        # 保存或更新 `end_point` 的值。
        self.end_point = self.generate_pos_obst_map()

        # Generate obstacle-free trajectory points
        # 保存或更新 `num_samples` 的值。
        num_samples = 10
        # 保存或更新 `max_dist` 的值。
        max_dist = 4.0
        # 保存或更新 `sampled_points_idx` 的值。
        sampled_points_idx = []
        # 在条件成立时持续执行下面的循环体。
        while len(sampled_points_idx) < num_samples:
            # Randomly select a point
            # 保存或更新 `point_idx` 的值。
            point_idx = np.random.choice(len(self.free_space))

            # Check if the distance constraint is satisfied with the previously sampled points
            # 根据条件决定是否进入当前分支。
            if len(sampled_points_idx) > 0:
                # 保存或更新 `distances` 的值。
                distances = np.array([np.linalg.norm(self.cell_centers[sampled_point_idx] - self.cell_centers[point_idx])
                                      for sampled_point_idx in sampled_points_idx])
                # 根据条件决定是否进入当前分支。
                if np.any(distances > max_dist):
                    # 跳过本轮循环剩余逻辑，进入下一轮。
                    continue

            # Add the point to the sampled trajectory and remove it from the free space
            # 调用 `append` 执行当前处理。
            sampled_points_idx.append(point_idx)
            # 调用 `pop` 执行当前处理。
            self.free_space.pop(point_idx)

        # Separate x and y coordinates of the sampled points
        # 保存或更新 `sampled_points` 的值。
        self.sampled_points = self.cell_centers[sampled_points_idx]

        # Reset formation and related parameters
        # 调用 `update_formation_and_relate_param` 执行当前处理。
        self.update_formation_and_relate_param()

        # Reassign goals
        # 保存或更新 `spawn_points` 的值。
        self.spawn_points = copy.deepcopy(self.start_point)
        # 保存或更新 `goals` 的值。
        self.goals = np.array([self.end_point for _ in range(self.num_agents)])
