import numpy as np
import copy
import bezier

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base

# 这是障碍环境里的 Bezier 轨迹场景。
# 它把无障碍 `ep_rand_bezier` 的“共享移动目标”思路迁移到 obstacle map 约束下：
# 起点先从自由格中为每架无人机独立采样，随后全队共享一个会沿 Bezier 曲线平滑移动的目标点。


class Scenario_o_ep_rand_bezier(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # 这个字段在当前实现里不是主导节奏的参数，真正轨迹窗口由 `num_secs` 控制。
        duration_time = 0.3
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        self.approch_goal_metric = 1.0

    def step(self):
        # 每 6 秒重采一条新的共享 Bezier 轨迹，并在该时间窗内按离散采样点平滑推进目标。
        tick = self.envs[0].tick
        control_freq = self.envs[0].control_freq
        num_secs = 6
        control_steps = int(num_secs * control_freq)
        t = tick % control_steps

        room_dims = np.array(self.room_dims) - self.formation_size
        # 在障碍版本里把轨迹步长上限压得更小，降低目标穿越障碍稠密区域时的突然跳变。
        max_dist = min(5, max(room_dims))
        min_dist = max_dist / 2
        if tick % control_steps == 0 or tick == 1:
            # 重采控制点直到整条短程运动仍处在允许的房间边界内。
            new_goal_found = False
            while not new_goal_found:
                low, high = np.array([-room_dims[0] / 2, -room_dims[1] / 2, 1.5]), np.array(
                    [room_dims[0] / 2, room_dims[1] / 2, 3.0])

                # 与普通版本类似，使用二阶 Bezier，所以需要两个未来控制点。
                new_pos = np.random.uniform(low=-high, high=high, size=(2, 3)).reshape(3, 2)
                new_pos = new_pos * np.random.randint(min_dist, max_dist + 1) / np.linalg.norm(new_pos, axis=0)
                new_pos = self.goals[0].reshape(3, 1) + new_pos
                lower_bound = np.expand_dims(low, axis=1)
                upper_bound = np.expand_dims(high, axis=1)
                new_goal_found = (new_pos > lower_bound + 0.5).all() and (
                    new_pos < upper_bound - 0.5).all()

            # 轨迹仍按“当前位置 + 两个未来控制点”离散化成整个窗口的目标序列。
            nodes = np.concatenate((self.goals[0].reshape(3, 1), new_pos), axis=1)
            nodes = np.asfortranarray(nodes)
            pts = np.linspace(0, 1, control_steps)
            curve = bezier.Curve(nodes, degree=2)
            self.interp = curve.evaluate_multi(pts)

        if tick % control_steps != 0 and tick > 1:
            # 当前实现让所有 agent 共享同一个瞬时 Bezier 目标点。
            self.goals = np.array([self.interp[:, t] for _ in range(self.num_agents)])

            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

        return

    def reset(self, obst_map=None, cell_centers=None):
        # 这里把旧的超短 duration 保留下来，但轨迹切换本身仍主要由 `step` 中的 6 秒窗口控制。
        self.duration_time = 0.01
        self.control_step_for_sec = int(self.duration_time * self.envs[0].control_freq)

        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None or cell_centers is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        # 出生点逐 agent 采样，但初始共享目标只采一个。
        self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        self.end_point = self.generate_pos_obst_map()

        # 下面这段会从自由格中采出一串障碍安全点并保存到 `sampled_points`。
        # 当前源码后续没有消费它，属于为轨迹约束预留的中间结果；注释副本保留实现，不改行为。
        num_samples = 10
        max_dist = 4.0
        sampled_points_idx = []
        while len(sampled_points_idx) < num_samples:
            point_idx = np.random.choice(len(self.free_space))

            if len(sampled_points_idx) > 0:
                distances = np.array([
                    np.linalg.norm(self.cell_centers[sampled_point_idx] - self.cell_centers[point_idx])
                    for sampled_point_idx in sampled_points_idx
                ])
                if np.any(distances > max_dist):
                    continue

            sampled_points_idx.append(point_idx)
            self.free_space.pop(point_idx)

        self.sampled_points = self.cell_centers[sampled_points_idx]

        self.update_formation_and_relate_param()

        self.spawn_points = copy.deepcopy(self.start_point)
        self.goals = np.array([self.end_point for _ in range(self.num_agents)])
