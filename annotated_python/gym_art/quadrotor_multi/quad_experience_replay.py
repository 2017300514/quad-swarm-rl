# 中文注释副本；原始文件：gym_art/quadrotor_multi/quad_experience_replay.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import random
from collections import deque
from copy import deepcopy

# 导入当前模块依赖。
import gymnasium as gym
import numpy as np


# 定义类 `ReplayBufferEvent`。
class ReplayBufferEvent:
    # 定义函数 `__init__`。
    def __init__(self, env, obs):
        # 保存或更新 `env` 的值。
        self.env = env
        # 保存或更新 `obs` 的值。
        self.obs = obs
        # 保存或更新 `num_replayed` 的值。
        self.num_replayed = 0


# 定义类 `ReplayBuffer`。
class ReplayBuffer:
    # 定义函数 `__init__`。
    def __init__(self, control_frequency, cp_step_size=0.5, buffer_size=20):
        # 保存或更新 `control_frequency` 的值。
        self.control_frequency = control_frequency
        # 保存或更新 `cp_step_size_sec` 的值。
        self.cp_step_size_sec = cp_step_size  # how often (seconds) a checkpoint is saved
        # 保存或更新 `cp_step_size_freq` 的值。
        self.cp_step_size_freq = self.cp_step_size_sec * self.control_frequency
        # 保存或更新 `buffer_idx` 的值。
        self.buffer_idx = 0
        # 保存或更新 `buffer` 的值。
        self.buffer = deque([], maxlen=buffer_size)

    # 定义函数 `write_cp_to_buffer`。
    def write_cp_to_buffer(self, env, obs):
        # 下面开始文档字符串说明。
        """
        A collision was found and we want to load the corresponding checkpoint from X seconds ago into the buffer to be sampled later on
        """
        # 保存或更新 `env.saved_in_replay_buffer` 的值。
        env.saved_in_replay_buffer = True

        # For example, replace the item with the lowest number of collisions in the last 10 replays
        # 保存或更新 `evt` 的值。
        evt = ReplayBufferEvent(env, obs)
        # 根据条件决定是否进入当前分支。
        if len(self.buffer) < self.buffer.maxlen:
            # 调用 `append` 执行当前处理。
            self.buffer.append(evt)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `buffer[buffer_idx]` 的值。
            self.buffer[self.buffer_idx] = evt
        # 调用 `print` 执行当前处理。
        print(f"Added new collision event to buffer at {self.buffer_idx}")
        # 保存或更新 `buffer_idx` 的值。
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer.maxlen

    # 定义函数 `sample_event`。
    def sample_event(self):
        # 下面开始文档字符串说明。
        """
        Sample an event to replay
        """
        # 保存或更新 `idx` 的值。
        idx = random.randint(0, len(self.buffer) - 1)
        # 调用 `print` 执行当前处理。
        print(f'Replaying event at idx {idx}')
        # 保存或更新 `buffer[idx].num_replayed` 的值。
        self.buffer[idx].num_replayed += 1
        # 返回当前函数的结果。
        return self.buffer[idx]

    # 定义函数 `cleanup`。
    def cleanup(self):
        # 保存或更新 `new_buffer` 的值。
        new_buffer = deque([], maxlen=self.buffer.maxlen)
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for event in self.buffer:
            # 根据条件决定是否进入当前分支。
            if event.num_replayed < 10:
                # 调用 `append` 执行当前处理。
                new_buffer.append(event)

        # 保存或更新 `buffer` 的值。
        self.buffer = new_buffer

    # 定义函数 `avg_num_replayed`。
    def avg_num_replayed(self):
        # 保存或更新 `replayed_stats` 的值。
        replayed_stats = [e.num_replayed for e in self.buffer]
        # 根据条件决定是否进入当前分支。
        if not replayed_stats:
            # 返回当前函数的结果。
            return 0
        # 返回当前函数的结果。
        return np.mean(replayed_stats)

    # 定义函数 `__len__`。
    def __len__(self):
        # 返回当前函数的结果。
        return len(self.buffer)


# 定义类 `ExperienceReplayWrapper`。
class ExperienceReplayWrapper(gym.Wrapper):
    # 定义函数 `__init__`。
    def __init__(self, env, replay_buffer_sample_prob, default_obst_density, defulat_obst_size,
                 domain_random=False, obst_density_random=False, obst_size_random=False,
                 # 保存或更新 `obst_density_min` 的值。
                 obst_density_min=0., obst_density_max=0., obst_size_min=0, obst_size_max=0.):
        # 调用 `super` 执行当前处理。
        super().__init__(env)
        # 保存或更新 `replay_buffer` 的值。
        self.replay_buffer = ReplayBuffer(env.envs[0].control_freq)
        # 保存或更新 `replay_buffer_sample_prob` 的值。
        self.replay_buffer_sample_prob = replay_buffer_sample_prob
        # 保存或更新 `curr_obst_density` 的值。
        self.curr_obst_density = default_obst_density
        # 保存或更新 `curr_obst_size` 的值。
        self.curr_obst_size = defulat_obst_size

        # 保存或更新 `domain_random` 的值。
        self.domain_random = domain_random
        # 根据条件决定是否进入当前分支。
        if self.domain_random:
            # 保存或更新 `obst_density_random` 的值。
            self.obst_density_random = obst_density_random
            # 保存或更新 `obst_size_random` 的值。
            self.obst_size_random = obst_size_random

            # 根据条件决定是否进入当前分支。
            if self.obst_density_random:
                # 保存或更新 `obst_densities` 的值。
                self.obst_densities = np.arange(obst_density_min, obst_density_max, 0.05)
                # 保存或更新 `curr_obst_density` 的值。
                self.curr_obst_density = 0.

            # 根据条件决定是否进入当前分支。
            if self.obst_size_random:
                # 保存或更新 `obst_sizes` 的值。
                self.obst_sizes = np.arange(obst_size_min, obst_size_max, 0.1)
                # 保存或更新 `curr_obst_size` 的值。
                self.curr_obst_size = 0.

        # 保存或更新 `max_episode_checkpoints_to_keep` 的值。
        self.max_episode_checkpoints_to_keep = int(3.0 / self.replay_buffer.cp_step_size_sec)  # keep only checkpoints from the last 3 seconds
        # 保存或更新 `episode_checkpoints` 的值。
        self.episode_checkpoints = deque([], maxlen=self.max_episode_checkpoints_to_keep)

        # 保存或更新 `save_time_before_collision_sec` 的值。
        self.save_time_before_collision_sec = 1.5
        # 保存或更新 `last_tick_added_to_buffer` 的值。
        self.last_tick_added_to_buffer = -1e9

        # variables for tensorboard
        # 保存或更新 `replayed_events` 的值。
        self.replayed_events = 0
        # 保存或更新 `episode_counter` 的值。
        self.episode_counter = 0

    # 定义函数 `save_checkpoint`。
    def save_checkpoint(self, obs):
        # 下面开始文档字符串说明。
        """
        Save a checkpoint every X steps so that we may load it later if a collision was found. This is NOT the same as the buffer
        Checkpoints are added to the buffer only if we find a collision and want to replay that event later on
        """
        # 调用 `append` 执行当前处理。
        self.episode_checkpoints.append((deepcopy(self.env), deepcopy(obs)))

    # 定义函数 `reset`。
    def reset(self):
        # 下面的文档字符串用于说明当前模块或代码块。
        """For reset we just use the default implementation."""
        # 保存或更新 `obst_density` 的值。
        obst_density = None
        # 保存或更新 `obst_size` 的值。
        obst_size = None
        # 根据条件决定是否进入当前分支。
        if self.domain_random:
            # 根据条件决定是否进入当前分支。
            if self.obst_density_random:
                # 保存或更新 `obst_density` 的值。
                obst_density = np.random.choice(self.obst_densities)
                # 保存或更新 `curr_obst_density` 的值。
                self.curr_obst_density = obst_density
            # 根据条件决定是否进入当前分支。
            if self.obst_size_random:
                # 保存或更新 `obst_size` 的值。
                obst_size = np.random.choice(self.obst_sizes)
                # 保存或更新 `curr_obst_size` 的值。
                self.curr_obst_size = obst_size

        # 返回当前函数的结果。
        return self.env.reset(obst_density, obst_size)

    # 定义函数 `step`。
    def step(self, action):
        # 同时更新 `obs`, `rewards`, `dones`, `infos` 等变量。
        obs, rewards, dones, infos = self.env.step(action)

        # 根据条件决定是否进入当前分支。
        if any(dones):
            # 保存或更新 `obs` 的值。
            obs = self.new_episode()
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i in range(len(infos)):
                # 根据条件决定是否进入当前分支。
                if not infos[i]["episode_extra_stats"]:
                    # 保存或更新 `infos[i][episode_extra_stats]` 的值。
                    infos[i]["episode_extra_stats"] = dict()

                # 保存或更新 `tag` 的值。
                tag = "replay"
                # 执行这一行逻辑。
                infos[i]["episode_extra_stats"].update({
                    f"{tag}/replay_rate": self.replayed_events / self.episode_counter,
                    f"{tag}/new_episode_rate": (self.episode_counter - self.replayed_events) / self.episode_counter,
                    f"{tag}/replay_buffer_size": len(self.replay_buffer),
                    f"{tag}/avg_replayed": self.replay_buffer.avg_num_replayed(),
                    f"{tag}/obst_density": self.curr_obst_density,
                    f"{tag}/obst_size": self.curr_obst_size,
                })

        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 根据条件决定是否进入当前分支。
            if self.env.use_replay_buffer and self.env.activate_replay_buffer and not self.env.saved_in_replay_buffer \
                    # 这里开始一个新的代码块。
                    and self.env.envs[0].tick % self.replay_buffer.cp_step_size_freq == 0:
                # 调用 `save_checkpoint` 执行当前处理。
                self.save_checkpoint(obs)

            # 保存或更新 `collision_flag` 的值。
            collision_flag = self.env.last_step_unique_collisions.any()
            # 根据条件决定是否进入当前分支。
            if self.env.use_obstacles:
                # 保存或更新 `collision_flag` 的值。
                collision_flag = collision_flag or len(self.env.curr_quad_col) > 0

            # 根据条件决定是否进入当前分支。
            if collision_flag and self.env.use_replay_buffer and self.env.activate_replay_buffer \
                    # 这里开始一个新的代码块。
                    and self.env.envs[0].tick > self.env.collisions_grace_period_seconds * self.env.envs[0].control_freq and not self.env.saved_in_replay_buffer:

                # 根据条件决定是否进入当前分支。
                if self.env.envs[0].tick - self.last_tick_added_to_buffer > 5 * self.env.envs[0].control_freq:
                    # added this check to avoid adding a lot of collisions from the same episode to the buffer

                    # 保存或更新 `steps_ago` 的值。
                    steps_ago = int(self.save_time_before_collision_sec / self.replay_buffer.cp_step_size_sec)
                    # 根据条件决定是否进入当前分支。
                    if steps_ago > len(self.episode_checkpoints):
                        # 调用 `print` 执行当前处理。
                        print(f"Tried to read past the boundary of checkpoint_history. Steps ago: {steps_ago}, episode checkpoints: {len(self.episode_checkpoints)}, {self.env.envs[0].tick}")
                        # 主动抛出异常以中止或提示错误。
                        raise IndexError
                    # 当前置条件都不满足时，执行兜底分支。
                    else:
                        # 同时更新 `env`, `obs` 等变量。
                        env, obs = self.episode_checkpoints[-steps_ago]
                        # 调用 `write_cp_to_buffer` 执行当前处理。
                        self.replay_buffer.write_cp_to_buffer(env, obs)
                        # 保存或更新 `env.collision_occurred` 的值。
                        self.env.collision_occurred = False  # this allows us to add a copy of this episode to the buffer once again if another collision happens

                        # 保存或更新 `last_tick_added_to_buffer` 的值。
                        self.last_tick_added_to_buffer = self.env.envs[0].tick

        # 返回当前函数的结果。
        return obs, rewards, dones, infos

    # 定义函数 `new_episode`。
    def new_episode(self):
        # 下面开始文档字符串说明。
        """
        Normally this would go into reset(), but MultiQuadEnv is a multi-agent env that automatically resets.
        This means that reset() is never actually called externally and we need to take care of starting our new episode.
        """
        # 保存或更新 `episode_counter` 的值。
        self.episode_counter += 1
        # 保存或更新 `last_tick_added_to_buffer` 的值。
        self.last_tick_added_to_buffer = -1e9
        # 保存或更新 `episode_checkpoints` 的值。
        self.episode_checkpoints = deque([], maxlen=self.max_episode_checkpoints_to_keep)

        # 根据条件决定是否进入当前分支。
        if np.random.uniform(0, 1) < self.replay_buffer_sample_prob and self.replay_buffer and self.env.activate_replay_buffer \
                # 这里开始一个新的代码块。
                and len(self.replay_buffer) > 0:
            # 保存或更新 `replayed_events` 的值。
            self.replayed_events += 1
            # 保存或更新 `event` 的值。
            event = self.replay_buffer.sample_event()
            # 保存或更新 `env` 的值。
            env = event.env
            # 保存或更新 `obs` 的值。
            obs = event.obs
            # 保存或更新 `replayed_env` 的值。
            replayed_env = deepcopy(env)
            # 保存或更新 `replayed_env.scenes` 的值。
            replayed_env.scenes = self.env.scenes
            # 保存或更新 `curr_obst_density` 的值。
            self.curr_obst_density = replayed_env.obst_density

            # we want to use these for tensorboard, so reset them to zero to get accurate stats
            # 保存或更新 `replayed_env.collisions_per_episode` 的值。
            replayed_env.collisions_per_episode = replayed_env.collisions_after_settle = 0
            # 保存或更新 `replayed_env.obst_quad_collisions_per_episode` 的值。
            replayed_env.obst_quad_collisions_per_episode = replayed_env.obst_quad_collisions_after_settle = 0
            # 保存或更新 `env` 的值。
            self.env = replayed_env

            # 调用 `cleanup` 执行当前处理。
            self.replay_buffer.cleanup()

            # 返回当前函数的结果。
            return obs

        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `obst_density` 的值。
            obst_density = None
            # 保存或更新 `obst_size` 的值。
            obst_size = None
            # 根据条件决定是否进入当前分支。
            if self.domain_random:
                # 根据条件决定是否进入当前分支。
                if self.obst_density_random:
                    # 保存或更新 `obst_density` 的值。
                    obst_density = np.random.choice(self.obst_densities)
                    # 保存或更新 `curr_obst_density` 的值。
                    self.curr_obst_density = obst_density
                # 根据条件决定是否进入当前分支。
                if self.obst_size_random:
                    # 保存或更新 `obst_size` 的值。
                    obst_size = np.random.choice(self.obst_sizes)
                    # 保存或更新 `curr_obst_size` 的值。
                    self.curr_obst_size = obst_size

            # 保存或更新 `obs` 的值。
            obs = self.env.reset(obst_density, obst_size)

            # 保存或更新 `env.saved_in_replay_buffer` 的值。
            self.env.saved_in_replay_buffer = False
            # 返回当前函数的结果。
            return obs
