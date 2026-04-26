# 中文注释副本；原始文件：gym_art/quadrotor_multi/tests/test_multi_env.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import time
from unittest import TestCase
import numpy as np

# 导入当前模块依赖。
from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti


# 定义函数 `create_env`。
def create_env(num_agents, use_numba=False, use_replay_buffer=False, episode_duration=7, local_obs=-1):
    # 保存或更新 `quad` 的值。
    quad = 'Crazyflie'
    # 保存或更新 `dyn_randomize_every` 的值。
    dyn_randomize_every = dyn_randomization_ratio = None

    # 保存或更新 `episode_duration` 的值。
    episode_duration = episode_duration  # seconds

    # 保存或更新 `raw_control` 的值。
    raw_control = raw_control_zero_middle = True

    # 保存或更新 `sampler_1` 的值。
    sampler_1 = None
    # 根据条件决定是否进入当前分支。
    if dyn_randomization_ratio is not None:
        # 保存或更新 `sampler_1` 的值。
        sampler_1 = dict(type="RelativeSampler", noise_ratio=dyn_randomization_ratio, sampler="normal")

    # 保存或更新 `sense_noise` 的值。
    sense_noise = 'default'

    # 保存或更新 `dynamics_change` 的值。
    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    # 保存或更新 `env` 的值。
    env = QuadrotorEnvMulti(
        num_agents=num_agents,
        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=True, ep_time=episode_duration, quads_use_numba=use_numba,
        use_replay_buffer=use_replay_buffer,
        swarm_obs="pos_vel_goals_ndist_gdist",
        local_obs=local_obs,
    )
    # 返回当前函数的结果。
    return env


# 定义类 `TestMultiEnv`。
class TestMultiEnv(TestCase):
    # 定义函数 `test_basic`。
    def test_basic(self):
        # 保存或更新 `num_agents` 的值。
        num_agents = 2
        # 保存或更新 `env` 的值。
        env = create_env(num_agents, use_numba=False)

        # 调用 `assertTrue` 执行当前处理。
        self.assertTrue(hasattr(env, 'num_agents'))
        # 调用 `assertEqual` 执行当前处理。
        self.assertEqual(env.num_agents, num_agents)

        # 保存或更新 `obs` 的值。
        obs = env.reset()
        # 调用 `assertIsNotNone` 执行当前处理。
        self.assertIsNotNone(obs)

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(100):
            # 同时更新 `obs`, `rewards`, `dones`, `infos` 等变量。
            obs, rewards, dones, infos = env.step([env.action_space.sample() for i in range(num_agents)])
            # 尝试执行下面的逻辑，并为异常情况做准备。
            try:
                # 调用 `assertIsInstance` 执行当前处理。
                self.assertIsInstance(obs, list)
            # 捕获前面代码可能抛出的异常。
            except:
                # 调用 `assertIsInstance` 执行当前处理。
                self.assertIsInstance(obs, np.ndarray)

            # 调用 `assertIsInstance` 执行当前处理。
            self.assertIsInstance(rewards, list)
            # 调用 `assertIsInstance` 执行当前处理。
            self.assertIsInstance(dones, list)
            # 调用 `assertIsInstance` 执行当前处理。
            self.assertIsInstance(infos, list)

        # 调用 `close` 执行当前处理。
        env.close()

    # 定义函数 `test_render`。
    def test_render(self):
        # 保存或更新 `num_agents` 的值。
        num_agents = 16
        # 保存或更新 `env` 的值。
        env = create_env(num_agents, use_numba=False, local_obs=8)
        # 保存或更新 `env.render_speed` 的值。
        env.render_speed = 1.0

        # 调用 `reset` 执行当前处理。
        env.reset()
        # 调用 `sleep` 执行当前处理。
        time.sleep(0.1)

        # 保存或更新 `num_steps` 的值。
        num_steps = 0
        # 保存或更新 `render_n_frames` 的值。
        render_n_frames = 100

        # 保存或更新 `render_start` 的值。
        render_start = None
        # 在条件成立时持续执行下面的循环体。
        while num_steps < render_n_frames:
            # 同时更新 `obs`, `rewards`, `dones`, `infos` 等变量。
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            # 保存或更新 `num_steps` 的值。
            num_steps += 1
            # print('Rewards: ', rewards, "\nCollisions: \n", env.collisions, "\nDistances: \n", env.dist)
            # 调用 `render` 执行当前处理。
            env.render()

            # 根据条件决定是否进入当前分支。
            if num_steps <= 1:
                # 保存或更新 `render_start` 的值。
                render_start = time.time()

        # 保存或更新 `render_took` 的值。
        render_took = time.time() - render_start
        # 调用 `print` 执行当前处理。
        print(f"Rendering of {render_n_frames} frames took {render_took:.3f} sec")

        # 调用 `close` 执行当前处理。
        env.close()

    # 定义函数 `test_local_info`。
    def test_local_info(self):
        # 保存或更新 `num_agents` 的值。
        num_agents = 16
        # 保存或更新 `env` 的值。
        env = create_env(num_agents, use_numba=False, local_obs=8)

        # 调用 `reset` 执行当前处理。
        env.reset()

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(100):
            # 同时更新 `obs`, `rewards`, `dones`, `infos` 等变量。
            obs, rewards, dones, infos = env.step([env.action_space.sample() for i in range(num_agents)])

        # 调用 `close` 执行当前处理。
        env.close()


# 定义类 `TestReplayBuffer`。
class TestReplayBuffer(TestCase):
    # 定义函数 `test_replay`。
    def test_replay(self):
        # 保存或更新 `num_agents` 的值。
        num_agents = 16
        # 保存或更新 `replay_buffer_sample_prob` 的值。
        replay_buffer_sample_prob = 1.0
        # 保存或更新 `env` 的值。
        env = create_env(num_agents, use_numba=False, use_replay_buffer=replay_buffer_sample_prob > 0, episode_duration=5)
        # 保存或更新 `env.render_speed` 的值。
        env.render_speed = 1.0
        # 保存或更新 `env` 的值。
        env = ExperienceReplayWrapper(env, replay_buffer_sample_prob=replay_buffer_sample_prob)

        # 调用 `reset` 执行当前处理。
        env.reset()
        # 调用 `sleep` 执行当前处理。
        time.sleep(0.01)

        # 保存或更新 `num_steps` 的值。
        num_steps = 0
        # 保存或更新 `render_n_frames` 的值。
        render_n_frames = 150

        # 在条件成立时持续执行下面的循环体。
        while num_steps < render_n_frames:
            # 同时更新 `obs`, `rewards`, `dones`, `infos` 等变量。
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            # 保存或更新 `num_steps` 的值。
            num_steps += 1
            # print('Rewards: ', rewards, "\nCollisions: \n", env.collisions, "\nDistances: \n", env.dist)
            # 调用 `render` 执行当前处理。
            env.render()
            # this env self-resets

        # 调用 `close` 执行当前处理。
        env.close()
