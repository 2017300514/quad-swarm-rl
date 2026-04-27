# 中文注释副本；原始文件：gym_art/quadrotor_multi/tests/test_multi_env.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这个文件是多机环境本体的基础测试入口。
# 它统一构造一个轻量 `QuadrotorEnvMulti`，然后分别检查基础 step、渲染、本地观测和 replay buffer 包装这几条常用链路。

import time
from unittest import TestCase
import numpy as np

from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti


# 这个 helper 把测试里反复出现的环境默认参数折叠起来。
# 重点不是造“真实训练配置”，而是造一个足够轻、足够稳定的测试环境。
def create_env(num_agents, use_numba=False, use_replay_buffer=False, episode_duration=7, local_obs=-1):
    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None

    episode_duration = episode_duration  # seconds

    # 测试里直接用原始电机控制，避免再引入额外控制器层的不确定性。
    raw_control = raw_control_zero_middle = True

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        # 如果打开动力学随机化，这里会给环境提供一个相对采样器；默认测试保持关闭，减少噪声来源。
        sampler_1 = dict(type="RelativeSampler", noise_ratio=dyn_randomization_ratio, sampler="normal")

    sense_noise = 'default'

    # 这里只保留最小必要的动力学改写，方便测试噪声和阻尼路径是否能正常走通。
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
    return env


# 这组测试覆盖多机环境本体的几条主链：step、render 和 local observation。
class TestMultiEnv(TestCase):
    # 最基础的冒烟测试：环境是否能 reset，step 后返回的数据结构是否还是预期形状。
    def test_basic(self):
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

    # 渲染测试主要覆盖 `env.render()` 和环境主循环能否在开启 local obs 的情况下稳定共存。
    def test_render(self):
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

    # 这里不看渲染，只单独让 `local_obs=8` 运行一段时间，确认邻居局部观测路径不会在长期 step 中出错。
    def test_local_info(self):
        num_agents = 16
        env = create_env(num_agents, use_numba=False, local_obs=8)

        env.reset()

        for i in range(100):
            obs, rewards, dones, infos = env.step([env.action_space.sample() for i in range(num_agents)])

        env.close()


# 这一组单独验证 replay wrapper 包上去之后，环境仍然能自重置、step 和 render。
class TestReplayBuffer(TestCase):
    # 这里把采样概率直接设成 1.0，是为了强制覆盖 replay buffer 路径，而不是测试真实训练推荐超参。
    def test_replay(self):
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
            env.render()
            # this env self-resets

        env.close()
