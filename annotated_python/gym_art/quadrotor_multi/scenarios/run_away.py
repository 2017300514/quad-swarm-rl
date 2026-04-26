import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario

# 这是一个带局部追逃味道的场景变体。
# reset 时先像普通不同目标任务一样生成整组 goals；step 中再周期性改写前两架无人机的目标，
# 让它们去追逐其余 agent 当前占据的目标点，从而制造局部“逃逸/追逐”扰动。


class Scenario_run_away(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)

    def update_goals(self):
        # 当外部通过可视化或交互接口改变 formation_size 时，需要先重建整组基准 goals，
        # 然后再由 step 的局部追逃逻辑继续覆盖前两架无人机的目标。
        self.goals = self.generate_goals(self.num_agents, self.formation_center, layer_dist=self.layer_dist)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self):
        tick = self.envs[0].tick
        # 这个场景按 1 秒粒度注入一次追逃扰动，节奏比换整组场景更快。
        control_step_for_sec = int(1.0 * self.envs[0].control_freq)

        if tick % control_step_for_sec == 0 and tick > 0:
            # 按当前实现，只改写前两架无人机的目标，并且它们只能从其余 agent 的目标里抽样。
            # 结果是大部分队形保持原位，局部出现更强的交叉追逐和碰撞压力。
            g_index = np.random.randint(low=1, high=self.num_agents, size=2)
            self.goals[0] = self.goals[g_index[0]]
            self.goals[1] = self.goals[g_index[1]]
            self.envs[0].goal = self.goals[0]
            self.envs[1].goal = self.goals[1]

        return

    def reset(self):
        # reset 仍沿用“先抽编队参数、再生成一组完整 goals”的主线，只是没有调用 `standard_reset` 封装。
        self.update_formation_and_relate_param()
        self.formation_center = np.array([0.0, 0.0, 2.0])

        # 这里只准备初始 goal 集合；真正把第 i 个 goal 写入第 i 个子环境，是外层多机环境 reset 的职责。
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.update_goals()
