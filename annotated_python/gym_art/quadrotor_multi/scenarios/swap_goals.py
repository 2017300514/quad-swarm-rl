import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario

# 这个场景保留同一组编队目标点，但会周期性交换“哪架无人机追哪个目标”。
# 上游由基类 reset 先生成一组固定 goals；下游 `quadrotor_multi.py` 在每步推进后读取这里写回的
# `env.goal`，从而把任务难点从“跟一个静态目标点”改成“目标集合不变但分配关系突变”。


class Scenario_swap_goals(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # 把“每隔几秒重新分配一次目标”换算成控制步数；reset 时会再次随机化这个节奏。
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_goals(self):
        # 与 `dynamic_diff_goal` 不同，这里不重采新的几何中心或编队形状，只在现有 goal 集合内部洗牌。
        # 这样场景难点集中在“agent-goal 映射突变”，而不是空间布局整体改变。
        np.random.shuffle(self.goals)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self):
        tick = self.envs[0].tick
        # 每到切换时刻，就把新映射同步回各子环境；下一步 reward 和 distance-to-goal 都会基于新目标结算。
        if tick % self.control_step_for_sec == 0 and tick > 0:
            self.update_goals()

        return

    def reset(self):
        # 每个 episode 随机换目标节奏，避免策略只适应固定的交换频率。
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # 初始 goal 集合仍复用基类标准流程：抽编队、定中心、生成并打乱一组目标点。
        self.standard_reset()
