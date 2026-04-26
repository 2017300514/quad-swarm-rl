#!/usr/bin/env python
# 中文注释副本；原始文件：gym_art/quadrotor_multi/quad_experience_replay.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件实现论文里的碰撞片段经验回放机制。
# 上游输入来自多机环境每一步产生的碰撞标记、障碍配置和当前观测；
# 下游输出是在新 episode 开始时，决定是正常 reset，还是从“碰撞前约 1.5 秒”的历史检查点继续训练。
# 这条链路的目标不是复用通用 RL replay，而是放大稀有但关键的失败事件，让策略反复练习高风险局面。

import random
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np


class ReplayBufferEvent:
    # 一个 event 就是一段可被重新开局的历史片段：
    # 包括完整环境快照和与之对应的观测，以及它已经被重放过多少次。
    def __init__(self, env, obs):
        self.env = env
        self.obs = obs
        self.num_replayed = 0


class ReplayBuffer:
    # 这个缓冲区不保存逐步 transition，而是保存“碰撞前某个时刻的整段环境检查点”。
    # 因此它更像失败事件库，而不是 DQN/离策略算法里的样本回放池。
    def __init__(self, control_frequency, cp_step_size=0.5, buffer_size=20):
        self.control_frequency = control_frequency
        self.cp_step_size_sec = cp_step_size  # how often (seconds) a checkpoint is saved
        self.cp_step_size_freq = self.cp_step_size_sec * self.control_frequency
        self.buffer_idx = 0
        self.buffer = deque([], maxlen=buffer_size)

    def write_cp_to_buffer(self, env, obs):
        """
        A collision was found and we want to load the corresponding checkpoint from X seconds ago into the buffer to be sampled later on
        """
        # 这个标志告诉环境：本局的关键碰撞已经存过一次，不要因为同一串连锁碰撞反复写入缓冲区。
        env.saved_in_replay_buffer = True

        # 这里采用简单的循环覆盖策略维护事件池。
        # 关注点不是“最优优先级采样”，而是持续保留一批近期独特碰撞片段可供重开。
        evt = ReplayBufferEvent(env, obs)
        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append(evt)
        else:
            self.buffer[self.buffer_idx] = evt
        print(f"Added new collision event to buffer at {self.buffer_idx}")
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer.maxlen

    def sample_event(self):
        """
        Sample an event to replay
        """
        # 新 episode 开始时，如果命中了 replay 概率，就随机抽一条历史碰撞片段重开。
        idx = random.randint(0, len(self.buffer) - 1)
        print(f'Replaying event at idx {idx}')
        self.buffer[idx].num_replayed += 1
        return self.buffer[idx]

    def cleanup(self):
        # 一条失败事件被反复重播太多次后就丢掉，避免训练长期被少数样本垄断。
        new_buffer = deque([], maxlen=self.buffer.maxlen)
        for event in self.buffer:
            if event.num_replayed < 10:
                new_buffer.append(event)

        self.buffer = new_buffer

    def avg_num_replayed(self):
        replayed_stats = [e.num_replayed for e in self.buffer]
        if not replayed_stats:
            return 0
        return np.mean(replayed_stats)

    def __len__(self):
        return len(self.buffer)


class ExperienceReplayWrapper(gym.Wrapper):
    # 这个 wrapper 把“正常多机环境”扩展成“偶尔从历史碰撞前片段重开”的环境。
    # 它夹在 `QuadrotorEnvMulti` 外层，一边定期存检查点，一边在满足条件时把碰撞前片段推进 replay buffer。
    def __init__(self, env, replay_buffer_sample_prob, default_obst_density, defulat_obst_size,
                 domain_random=False, obst_density_random=False, obst_size_random=False,
                 obst_density_min=0., obst_density_max=0., obst_size_min=0, obst_size_max=0.):
        super().__init__(env)
        self.replay_buffer = ReplayBuffer(env.envs[0].control_freq)
        self.replay_buffer_sample_prob = replay_buffer_sample_prob
        self.curr_obst_density = default_obst_density
        self.curr_obst_size = defulat_obst_size

        self.domain_random = domain_random
        if self.domain_random:
            self.obst_density_random = obst_density_random
            self.obst_size_random = obst_size_random

            if self.obst_density_random:
                self.obst_densities = np.arange(obst_density_min, obst_density_max, 0.05)
                self.curr_obst_density = 0.

            if self.obst_size_random:
                self.obst_sizes = np.arange(obst_size_min, obst_size_max, 0.1)
                self.curr_obst_size = 0.

        # 每个 episode 只保留最近 3 秒的环境快照。
        # 这是因为本机制关注的是“碰撞前短时间内的决策错误”，而不是很早之前的整局历史。
        self.max_episode_checkpoints_to_keep = int(3.0 / self.replay_buffer.cp_step_size_sec)  # keep only checkpoints from the last 3 seconds
        self.episode_checkpoints = deque([], maxlen=self.max_episode_checkpoints_to_keep)

        # 当识别到独特碰撞时，实际写入 buffer 的是“碰撞前 1.5 秒”的检查点，而不是碰撞瞬间。
        # 这样策略重放后还有时间重新做出规避动作。
        self.save_time_before_collision_sec = 1.5
        self.last_tick_added_to_buffer = -1e9

        # variables for tensorboard
        self.replayed_events = 0
        self.episode_counter = 0

    def save_checkpoint(self, obs):
        """
        Save a checkpoint every X steps so that we may load it later if a collision was found. This is NOT the same as the buffer
        Checkpoints are added to the buffer only if we find a collision and want to replay that event later on
        """
        # 这里保存的是“本局内的候选回溯点”。
        # 只有后面真的观察到碰撞时，才会从这些检查点里抽取一个写进 replay buffer。
        self.episode_checkpoints.append((deepcopy(self.env), deepcopy(obs)))

    def reset(self):
        """For reset we just use the default implementation."""
        # 外部首次 reset 仍然走正常环境 reset。
        # 如果启用了障碍 domain randomization，就在这里同步抽本局障碍密度和障碍尺寸。
        obst_density = None
        obst_size = None
        if self.domain_random:
            if self.obst_density_random:
                obst_density = np.random.choice(self.obst_densities)
                self.curr_obst_density = obst_density
            if self.obst_size_random:
                obst_size = np.random.choice(self.obst_sizes)
                self.curr_obst_size = obst_size

        return self.env.reset(obst_density, obst_size)

    def step(self, action):
        # 先让底层环境正常推进。
        # 这个 wrapper 的核心职责不是改奖励，而是在 step 后观察是否该存 checkpoint、是否该提取失败片段。
        obs, rewards, dones, infos = self.env.step(action)

        if any(dones):
            # 多机环境会在内部自动开新局，因此这里要手动接管“episode 结束后的下一局从哪开始”。
            obs = self.new_episode()
            for i in range(len(infos)):
                if not infos[i]["episode_extra_stats"]:
                    infos[i]["episode_extra_stats"] = dict()

                tag = "replay"
                # 这些统计主要服务于 tensorboard，帮助判断当前训练有多少局来自 replay，而不是新随机场景。
                infos[i]["episode_extra_stats"].update({
                    f"{tag}/replay_rate": self.replayed_events / self.episode_counter,
                    f"{tag}/new_episode_rate": (self.episode_counter - self.replayed_events) / self.episode_counter,
                    f"{tag}/replay_buffer_size": len(self.replay_buffer),
                    f"{tag}/avg_replayed": self.replay_buffer.avg_num_replayed(),
                    f"{tag}/obst_density": self.curr_obst_density,
                    f"{tag}/obst_size": self.curr_obst_size,
                })

        else:
            if self.env.use_replay_buffer and self.env.activate_replay_buffer and not self.env.saved_in_replay_buffer \
                    and self.env.envs[0].tick % self.replay_buffer.cp_step_size_freq == 0:
                # 按固定时间步长记录候选检查点，供稍后回溯到“碰撞前若干秒”使用。
                self.save_checkpoint(obs)

            collision_flag = self.env.last_step_unique_collisions.any()
            if self.env.use_obstacles:
                # 障碍场景下，任一无人机撞障碍也会被视作可回放的失败事件。
                collision_flag = collision_flag or len(self.env.curr_quad_col) > 0

            if collision_flag and self.env.use_replay_buffer and self.env.activate_replay_buffer \
                    and self.env.envs[0].tick > self.env.collisions_grace_period_seconds * self.env.envs[0].control_freq and not self.env.saved_in_replay_buffer:

                if self.env.envs[0].tick - self.last_tick_added_to_buffer > 5 * self.env.envs[0].control_freq:
                    # 同一局里做一个冷却窗口，避免连续多步接触被误当成很多独立失败样本。

                    steps_ago = int(self.save_time_before_collision_sec / self.replay_buffer.cp_step_size_sec)
                    if steps_ago > len(self.episode_checkpoints):
                        print(f"Tried to read past the boundary of checkpoint_history. Steps ago: {steps_ago}, episode checkpoints: {len(self.episode_checkpoints)}, {self.env.envs[0].tick}")
                        raise IndexError
                    else:
                        # 真正写进 replay buffer 的是“碰撞前 1.5 秒”的历史快照。
                        # 这正对应论文里从碰撞前裁剪片段继续训练的做法。
                        env, obs = self.episode_checkpoints[-steps_ago]
                        self.replay_buffer.write_cp_to_buffer(env, obs)
                        self.env.collision_occurred = False  # this allows us to add a copy of this episode to the buffer once again if another collision happens

                        self.last_tick_added_to_buffer = self.env.envs[0].tick

        return obs, rewards, dones, infos

    def new_episode(self):
        """
        Normally this would go into reset(), but MultiQuadEnv is a multi-agent env that automatically resets.
        This means that reset() is never actually called externally and we need to take care of starting our new episode.
        """
        # 每次进入新局前，先清本局 checkpoint 历史和节流状态。
        self.episode_counter += 1
        self.last_tick_added_to_buffer = -1e9
        self.episode_checkpoints = deque([], maxlen=self.max_episode_checkpoints_to_keep)

        if np.random.uniform(0, 1) < self.replay_buffer_sample_prob and self.replay_buffer and self.env.activate_replay_buffer \
                and len(self.replay_buffer) > 0:
            # 命中 replay 时，不做随机 reset，而是直接把历史碰撞前的环境快照接管成当前环境。
            self.replayed_events += 1
            event = self.replay_buffer.sample_event()
            env = event.env
            obs = event.obs
            replayed_env = deepcopy(env)
            replayed_env.scenes = self.env.scenes
            self.curr_obst_density = replayed_env.obst_density

            # 回放开局后，本局碰撞统计要从 0 重新累计，否则 tensorboard 会把历史片段里的旧碰撞也算进当前局。
            replayed_env.collisions_per_episode = replayed_env.collisions_after_settle = 0
            replayed_env.obst_quad_collisions_per_episode = replayed_env.obst_quad_collisions_after_settle = 0
            self.env = replayed_env

            self.replay_buffer.cleanup()

            return obs

        else:
            # 没命中 replay 时，按正常随机环境开新局，并继续支持障碍 domain randomization。
            obst_density = None
            obst_size = None
            if self.domain_random:
                if self.obst_density_random:
                    obst_density = np.random.choice(self.obst_densities)
                    self.curr_obst_density = obst_density
                if self.obst_size_random:
                    obst_size = np.random.choice(self.obst_sizes)
                    self.curr_obst_size = obst_size

            obs = self.env.reset(obst_density, obst_size)

            self.env.saved_in_replay_buffer = False
            return obs
