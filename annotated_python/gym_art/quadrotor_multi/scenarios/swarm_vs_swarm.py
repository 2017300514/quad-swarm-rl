import copy
import numpy as np

from gym_art.quadrotor_multi.scenarios.utils import get_z_value
from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario

# 这个场景把整队无人机拆成两个子群，各自维护一套编队目标，然后周期性交换两群的目标中心。
# 上游仍是基类提供的编队几何工具；下游 `quadrotor_multi.py` 会把这里拼出的 `self.goals`
# 分配给所有 agent，并按场景名分别统计 distance-to-goal 与碰撞指标。


class Scenario_swarm_vs_swarm(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # 两个子群不是每步都改目标，而是每隔几秒整体交换一次目标中心。
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        self.goals_1, self.goals_2 = None, None
        self.goal_center_1, self.goal_center_2 = None, None

    def formation_centers(self):
        if self.formation_center is None:
            self.formation_center = np.array([0., 0., 2.])

        box_size = self.envs[0].box
        dist_low_bound = self.lowest_formation_size
        # 先在房间内随机放置第一群目标中心；z 下界仍依赖编队尺寸，避免目标组贴地。
        x, y = np.random.uniform(low=-box_size, high=box_size, size=(2,))
        z = get_z_value(num_agents=self.num_agents, num_agents_per_layer=self.num_agents_per_layer,
                        box_size=box_size, formation=self.formation, formation_size=self.formation_size)

        goal_center_1 = np.array([x, y, z])

        # 第二群中心按随机方位和随机距离相对第一群偏移，制造真正的“双群对向/对抗”空间关系。
        goal_center_distance = np.random.uniform(low=box_size / 4, high=box_size)

        phi = np.random.uniform(low=-np.pi, high=np.pi)
        theta = np.random.uniform(low=-0.5 * np.pi, high=0.5 * np.pi)
        goal_center_2 = goal_center_1 + goal_center_distance * np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        diff_x, diff_y, diff_z = goal_center_2 - goal_center_1
        # 这里额外保证两群在“编队法向方向”上不要离得过近，否则两套编队刚生成就可能互相重叠。
        if self.formation.endswith("horizontal"):
            if abs(diff_z) < dist_low_bound:
                goal_center_2[2] = np.sign(diff_z) * dist_low_bound + goal_center_1[2]
        elif self.formation.endswith("vertical_xz"):
            if abs(diff_y) < dist_low_bound:
                goal_center_2[1] = np.sign(diff_y) * dist_low_bound + goal_center_1[1]
        elif self.formation.endswith("vertical_yz"):
            if abs(diff_x) < dist_low_bound:
                goal_center_2[0] = np.sign(diff_x) * dist_low_bound + goal_center_1[0]

        return goal_center_1, goal_center_2

    def create_formations(self, goal_center_1, goal_center_2):
        # 两个子群分别按“半数 agent / 剩余 agent”生成各自编队。
        # 这里没有跨群混排，后续 concat 的顺序就是环境里 agent 编号收到目标的顺序。
        self.goals_1 = self.generate_goals(num_agents=self.num_agents // 2, formation_center=goal_center_1,
                                           layer_dist=self.layer_dist)
        self.goals_2 = self.generate_goals(num_agents=self.num_agents - self.num_agents // 2,
                                           formation_center=goal_center_2, layer_dist=self.layer_dist)
        self.goals = np.concatenate([self.goals_1, self.goals_2])

    def update_goals(self):
        # 核心事件不是随机重采两个新中心，而是直接交换两群当前的目标中心。
        # 这会强迫两支 swarm 穿过彼此原来的占位区域，形成更强的交汇与避碰压力。
        tmp_goal_center_1 = copy.deepcopy(self.goal_center_1)
        tmp_goal_center_2 = copy.deepcopy(self.goal_center_2)
        self.goal_center_1 = tmp_goal_center_2
        self.goal_center_2 = tmp_goal_center_1

        # 按当前实现，交换中心后还会重采一次 formation / formation_size / layer_dist，
        # 因此难点不只是“交换阵地”，还叠加了编队几何重新采样。
        self.update_formation_and_relate_param()
        self.create_formations(self.goal_center_1, self.goal_center_2)

        # 组内再打乱一次，让同一子群里具体哪架机去哪个槽位也不固定。
        np.random.shuffle(self.goals_1)
        np.random.shuffle(self.goals_2)
        self.goals = np.concatenate([self.goals_1, self.goals_2])
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

    def step(self):
        tick = self.envs[0].tick
        # 目标中心交换发生在 episode 中途，reward 与碰撞统计会立即切换到新的双群目标布局。
        if tick % self.control_step_for_sec == 0 and tick > 0:
            self.update_goals()
        return

    def reset(self):
        # 每局随机一个交换节奏，避免策略把对抗切换时机记成固定脚本。
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        self.update_formation_and_relate_param()

        # reset 时先为两群选中心，再分别生成两套编队目标。
        self.goal_center_1, self.goal_center_2 = self.formation_centers()
        self.create_formations(self.goal_center_1, self.goal_center_2)

        # `formation_center` 在这里退化成两个子群中心的中点，主要给上层共享接口、可视化和障碍初始化使用。
        self.formation_center = (self.goal_center_1 + self.goal_center_2) / 2

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            # 外部拖动编队尺寸时，不改两群中心，只在原位置重建两套子编队。
            self.create_formations(self.goal_center_1, self.goal_center_2)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]
