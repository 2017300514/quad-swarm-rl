# 中文注释副本；原始文件：gym_art/quadrotor_multi/tests/test_multi_env.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件属于多机四旋翼仿真环境的一部分，负责环境状态、物理过程或配套工具中的某一环。
# 它的上游通常来自场景配置、动力学状态或训练动作，下游会流向观测构造、奖励结算、碰撞处理或可视化。

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
import time
from unittest import TestCase
import numpy as np

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti


# `create_env` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def create_env(num_agents, use_numba=False, use_replay_buffer=False, episode_duration=7, local_obs=-1):
    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None

    episode_duration = episode_duration  # seconds

    raw_control = raw_control_zero_middle = True

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        # 这里构造的是环境默认奖励权重表，表示在没有实验覆盖时多机导航任务各个目标项的基准权重。
        sampler_1 = dict(type="RelativeSampler", noise_ratio=dyn_randomization_ratio, sampler="normal")

    sense_noise = 'default'

    # 这里构造的是环境默认奖励权重表，表示在没有实验覆盖时多机导航任务各个目标项的基准权重。
    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    env = QuadrotorEnvMulti(
        num_agents=num_agents,
        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=True, ep_time=episode_duration, quads_use_numba=use_numba,
        use_replay_buffer=use_replay_buffer,
        swarm_obs="pos_vel_goals_ndist_gdist",
        local_obs=local_obs,
    )
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return env


# `TestMultiEnv` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class TestMultiEnv(TestCase):
    # `test_basic` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def test_basic(self):
        # 该值来自实验配置，决定环境一次并行维护多少架无人机；后续会影响观测拼接尺寸、邻居筛选范围和碰撞矩阵规模。
        num_agents = 2
        env = create_env(num_agents, use_numba=False)

        self.assertTrue(hasattr(env, 'num_agents'))
        self.assertEqual(env.num_agents, num_agents)

        obs = env.reset()
        self.assertIsNotNone(obs)

        for i in range(100):
            obs, rewards, dones, infos = env.step([env.action_space.sample() for i in range(num_agents)])
            try:
                self.assertIsInstance(obs, list)
            except:
                self.assertIsInstance(obs, np.ndarray)

            self.assertIsInstance(rewards, list)
            self.assertIsInstance(dones, list)
            self.assertIsInstance(infos, list)

        env.close()

    # `test_render` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def test_render(self):
        # 该值来自实验配置，决定环境一次并行维护多少架无人机；后续会影响观测拼接尺寸、邻居筛选范围和碰撞矩阵规模。
        num_agents = 16
        env = create_env(num_agents, use_numba=False, local_obs=8)
        env.render_speed = 1.0

        env.reset()
        time.sleep(0.1)

        num_steps = 0
        render_n_frames = 100

        render_start = None
        while num_steps < render_n_frames:
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            num_steps += 1
            # print('Rewards: ', rewards, "\nCollisions: \n", env.collisions, "\nDistances: \n", env.dist)
            env.render()

            if num_steps <= 1:
                render_start = time.time()

        render_took = time.time() - render_start
        print(f"Rendering of {render_n_frames} frames took {render_took:.3f} sec")

        env.close()

    # `test_local_info` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def test_local_info(self):
        # 该值来自实验配置，决定环境一次并行维护多少架无人机；后续会影响观测拼接尺寸、邻居筛选范围和碰撞矩阵规模。
        num_agents = 16
        env = create_env(num_agents, use_numba=False, local_obs=8)

        env.reset()

        for i in range(100):
            obs, rewards, dones, infos = env.step([env.action_space.sample() for i in range(num_agents)])

        env.close()


# `TestReplayBuffer` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class TestReplayBuffer(TestCase):
    # `test_replay` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def test_replay(self):
        # 该值来自实验配置，决定环境一次并行维护多少架无人机；后续会影响观测拼接尺寸、邻居筛选范围和碰撞矩阵规模。
        num_agents = 16
        replay_buffer_sample_prob = 1.0
        env = create_env(num_agents, use_numba=False, use_replay_buffer=replay_buffer_sample_prob > 0, episode_duration=5)
        env.render_speed = 1.0
        env = ExperienceReplayWrapper(env, replay_buffer_sample_prob=replay_buffer_sample_prob)

        env.reset()
        time.sleep(0.01)

        num_steps = 0
        render_n_frames = 150

        while num_steps < render_n_frames:
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            num_steps += 1
            # print('Rewards: ', rewards, "\nCollisions: \n", env.collisions, "\nDistances: \n", env.dist)
            env.render()
            # this env self-resets

        env.close()
