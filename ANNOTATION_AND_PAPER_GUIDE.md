# 项目中文注释工作流与论文复现源码说明

## 1. 当前状态

- 记录更新时间：`2026-04-26`
- 注释副本目录：`annotated_python/`
- 原则：只修改 `annotated_python/` 内的注释副本，不修改原始源码。
- 当前结论：此前基于脚本模板批量生成的注释，不作为最终交付版本。
- 原因：模板生成只能按代码外形匹配，无法稳定解释该项目里变量的真实来源、状态流向、奖励意义、观测拼接逻辑和论文机制对应关系。

## 2. 后续采用的正式工作流

### 2.1 总原则

- 注释必须来自人工阅读源码后的理解，不能来自“变量赋值/函数调用/分支判断”这类语法模板改写。
- 注释的目标不是翻译代码字面动作，而是解释这段代码在四旋翼强化学习系统中的工作职责。
- 每条关键注释至少回答下列问题中的两个：
  - 数据从哪里来。
  - 这里为什么要改它。
  - 改完后谁会继续使用它。
  - 这段逻辑和训练、仿真、碰撞、观测、奖励、回放、部署中的哪一环有关。

### 2.2 单文件工作流程

对每一个文件，严格按下面顺序处理：

1. 先读原始源码，不直接改注释副本。  
   目标是先建立对该文件职责的完整判断，避免“边看边猜”。

2. 确认文件在项目中的角色。  
   判断它属于训练入口、环境主循环、单机动力学、障碍物、碰撞、场景、奖励包装、模型编码器、论文画图还是 sim2real。

3. 追踪关键输入来源。  
   要明确配置来自哪里、状态来自哪里、观测由哪些部分拼接、奖励由哪些项构成、统计量由哪些事件触发。

4. 追踪关键输出去向。  
   要明确本文件产出的状态、观测、奖励、模型特征、日志或导出结果会被哪个下游模块继续使用。

5. 提炼本文件的“真实主线”。  
   不是按代码顺序机械加注释，而是先总结这个文件真正解决的问题，再决定哪些位置值得写注释。

6. 只在关键位置落注。  
   文件头、核心类、核心函数、首次出现的重要状态变量、关键代码块、论文关键机制处必须注释；样板代码和低信息密度片段可以不注释。

7. 回读注释副本。  
   检查注释是否真的解释了项目逻辑，而不是把代码改写成中文废话。

### 2.3 每个文件必须完成的五项内容

1. 文件头说明  
   必须交代该文件在整个项目中的职责、上游输入、下游消费者。

2. 核心对象说明  
   对主要类或主要函数说明它维护什么状态，或者完成哪段关键流程。

3. 关键变量说明  
   只解释重要变量，但必须解释清楚来源、当前语义、后续用途。

4. 关键代码块说明  
   对观测拼接、奖励结算、邻居筛选、碰撞处理、障碍 SDF、回放缓存、注意力编码等代码块写块级说明。

5. 论文/实验语境说明  
   如果该文件直接对应论文机制，必须指出它和论文中的哪一部分对应。

## 3. 注释写作标准

### 3.1 允许的注释内容

- 解释状态来源：例如状态来自 `QuadrotorSingle`、场景生成器、障碍地图、Sample Factory 配置、历史 replay 片段。
- 解释状态变化原因：例如为了把秒级时间窗折算成 step、为了限制邻居观测槽位、为了在训练早期屏蔽初始化碰撞噪声。
- 解释后续用途：例如用于拼接策略输入、用于计算碰撞惩罚、用于写入日志、用于恢复 replay episode、用于导出 sim2real 模型。
- 解释结构设计：例如为什么 self / neighbor / obstacle 要分开编码，为什么 obstacle 观测固定成 9 维，为什么碰撞前 1.5 秒需要被截入回放。

### 3.2 禁止的注释内容

- “导入当前模块依赖”
- “定义函数/类”
- “调用某函数执行当前处理”
- “保存或更新某变量的值”
- “根据条件决定是否进入分支”
- 任何只是在中文里复述 Python 语法、但没有增加项目理解的信息

### 3.3 注释密度要求

- 不追求逐行全注释。
- 优先写高价值块注释，而不是给每一行都加一句空话。
- 同一个变量首次解释清楚后，后文除非语义变化，否则不重复解释。
- 遇到简单 `import`、样板测试、显然的参数透传，可以少注释或不注释。

## 4. 文件处理顺序

后续人工注释按“先主链路、后外围模块”的顺序推进：

1. `swarm_rl/train.py`
2. `swarm_rl/env_wrappers/quadrotor_params.py`
3. `swarm_rl/env_wrappers/quad_utils.py`
4. `swarm_rl/env_wrappers/reward_shaping.py`
5. `gym_art/quadrotor_multi/quadrotor_multi.py`
6. `gym_art/quadrotor_multi/quadrotor_single.py`
7. `gym_art/quadrotor_multi/quadrotor_dynamics.py`
8. `gym_art/quadrotor_multi/quadrotor_control.py`
9. `gym_art/quadrotor_multi/obstacles/obstacles.py`
10. `gym_art/quadrotor_multi/obstacles/utils.py`
11. `gym_art/quadrotor_multi/collisions/obstacles.py`
12. `gym_art/quadrotor_multi/collisions/quadrotors.py`
13. `gym_art/quadrotor_multi/quad_experience_replay.py`
14. `swarm_rl/models/quad_multi_model.py`
15. `swarm_rl/models/attention_layer.py`
16. 其余场景、测试、可视化、画图、sim2real 文件

这样做的原因是：先把训练入口、环境主循环、观测奖励链路和模型主干吃透，后面的场景文件和分析文件才能写得准。

## 5. 交付方式

- 不再使用脚本批量生成“全仓完成版”。
- 改为按文件或按模块逐批提交。
- 每完成一批，记录：
  - 处理了哪些文件
  - 注释覆盖了哪些关键机制
  - 还剩哪些模块未处理
  - 有没有需要回头补充上下文的文件

### 5.1 当前人工处理进度

- 已按人工流程重写：
  - `annotated_python/swarm_rl/train.py`
  - `annotated_python/swarm_rl/env_wrappers/quadrotor_params.py`
  - `annotated_python/swarm_rl/env_wrappers/quad_utils.py`
  - `annotated_python/swarm_rl/env_wrappers/reward_shaping.py`
  - `annotated_python/gym_art/quadrotor_multi/quadrotor_single.py`
  - `annotated_python/gym_art/quadrotor_multi/quadrotor_multi.py`
  - `annotated_python/gym_art/quadrotor_multi/quadrotor_dynamics.py`
  - `annotated_python/gym_art/quadrotor_multi/quadrotor_control.py`
  - `annotated_python/gym_art/quadrotor_multi/obstacles/obstacles.py`
  - `annotated_python/gym_art/quadrotor_multi/obstacles/utils.py`
  - `annotated_python/gym_art/quadrotor_multi/collisions/obstacles.py`
  - `annotated_python/gym_art/quadrotor_multi/collisions/quadrotors.py`
  - `annotated_python/gym_art/quadrotor_multi/quad_experience_replay.py`
  - `annotated_python/swarm_rl/models/quad_multi_model.py`
  - `annotated_python/swarm_rl/models/attention_layer.py`
  - `annotated_python/gym_art/quadrotor_multi/sensor_noise.py`
  - `annotated_python/gym_art/quadrotor_multi/collisions/utils.py`
  - `annotated_python/swarm_rl/sim2real/code_blocks.py`
  - `annotated_python/swarm_rl/sim2real/sim2real.py`
  - `annotated_python/swarm_rl/sim2real/torch_models/__init__.py`
  - `annotated_python/swarm_rl/sim2real/c_models/__init__.py`
  - `annotated_python/swarm_rl/sim2real/tests/unit_tests.py`
  - `annotated_python/gym_art/quadrotor_multi/get_state.py`
  - `annotated_python/gym_art/quadrotor_multi/collisions/room.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/base.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/utils.py`
  - `annotated_python/gym_art/quadrotor_multi/quadrotor_randomization.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/mix.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_same_goal.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/static_same_goal.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/swap_goals.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/run_away.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/swarm_vs_swarm.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_diff_goal.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_formations.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/static_diff_goal.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/ep_rand_bezier.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/ep_lissajous3D.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/__init__.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_base.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_random.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/test/o_test.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_static_same_goal.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_dynamic_same_goal.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_swap_goals.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_ep_rand_bezier.py`
  - `annotated_python/gym_art/quadrotor_multi/scenarios/test/__init__.py`
  - `annotated_python/paper/attn_heatmap.py`
  - `annotated_python/paper/fps_compare.py`
  - `annotated_python/paper/mean_std_plots_quad_obstacle.py`
- 本批重点覆盖：
  - 训练入口如何把环境注册、模型注册和两阶段配置解析接到 Sample Factory
  - 四旋翼实验参数如何分别流向观测、奖励、障碍物、回放、场景和 sim2real 链路
  - 环境构造函数如何把配置落实为 wrapper 链
  - reward shaping 如何改写 episode 级统计、true reward 和碰撞惩罚退火
  - 单机环境如何定义基础奖励、观测空间、动力学重采样和 reset/step
  - 多机环境如何聚合邻居观测、障碍物、碰撞修正、场景推进和 episode 级成功/失败指标
  - 动力学层如何把电机命令经过电机滞后、推力/力矩计算、姿态积分和地面/房间接触变成下一时刻状态
  - 控制层如何把 raw action、简化控制接口、速度/位置控制目标统一映射成四电机推力
  - 障碍物模块如何把场景中的障碍中心坐标转换成 9 维局部 SDF 风格观测，并输出无人机-障碍碰撞索引
  - 障碍碰撞和机间碰撞模块如何把碰撞法向、速度修正、角速度扰动和接近惩罚接回环境主循环
  - replay 机制如何从碰撞前短时间检查点重开 episode，以及模型主干如何融合 self / neighbor / obstacle 三段观测
  - 注意力算子本身和传感噪声链路如何支撑模型融合与 sim2real 观测逼真度
  - 碰撞修正底层工具和 sim2real C 模板代码如何把训练模型结构继续落到部署实现
  - sim2real 主入口如何把训练模型重建、抽权重并导出成部署源码
  - sim2real 测试链如何验证 PyTorch 模型与导出 C 模型在中间层和最终推力上保持一致
  - 自观测生成链和房间边界碰撞修正如何把动力学状态接回策略输入与碰撞统计
  - 场景基类和工具函数如何把高层任务模式翻译成编队几何、目标点集合和 reset/step 共享状态
  - `mix` 分发器如何在多任务训练中随机挑选真实 scenario，并把 goal / formation_size / spawn_points 回传外层环境
  - 同目标静态/动态场景、目标交换场景、局部追逃场景和双群对抗场景分别怎样改变 agent-goal 映射
  - 不同目标动态场景、编队伸缩场景和不同目标静态基线分别怎样改变目标集合的时间结构
  - 两种追逃轨迹场景怎样分别通过随机 Bezier 曲线和固定 Lissajous 周期轨迹驱动共享移动目标
  - 障碍物场景基类怎样把 obstacle map 转成自由栅格采样、spawn_points 和逐 agent 终点任务
  - 障碍同目标静态/动态场景和障碍目标交换场景怎样把普通场景逻辑迁移到 obstacle map 约束下
  - 障碍环境里的 Bezier 轨迹场景怎样把共享移动目标迁移到 obstacle map 约束下
  - 论文分析脚本怎样把硬编码分析矩阵、benchmark 数据和 TensorBoard 日志重新组织成最终论文图
- 下一批计划：
  - `annotated_python/paper/mean_std_plots_quad_compare_arch.py`
  - `annotated_python/paper/mean_std_plots_quad_obstacle_compare_arch_density.py`
  - `annotated_python/paper/mean_std_plots_quad_obstacle_compare_arch_neighbor.py`

## 6. 论文《Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning》对应源码说明

### 6.1 训练入口与实验配置

- `swarm_rl/train.py`：训练主入口。负责注册 `quadrotor_multi` 环境、注册自定义模型，并调用 Sample Factory 的 `run_rl` 开始 APPO/IPPO 训练流程。
- `swarm_rl/env_wrappers/quadrotor_params.py`：集中定义论文实验会用到的命令行参数，包括无人机数量、邻居观测、障碍物观测、回放概率、碰撞奖励、房间大小等。
- `swarm_rl/runs/obstacles/quad_obstacle_baseline.py`：障碍环境基线配置，给出论文主实验的默认超参数，包括 `replay_buffer_sample_prob=0.75`、`quads_use_obstacles=True`、`quads_obstacle_obs_type=octomap`、`quads_use_downwash=True` 等。
- `swarm_rl/runs/obstacles/quads_multi_obstacles.py`：论文最终多机障碍实验配置。在基线之上启用 `quads_neighbor_visible_num=2`、`quads_neighbor_obs_type=pos_vel` 和 `quads_encoder_type=attention`，对应论文里的邻居感知和注意力网络。

### 6.2 环境、状态、动作与奖励

- `gym_art/quadrotor_multi/quadrotor_multi.py`：多机环境核心实现。负责多架无人机实例化、邻居选择、障碍生成、碰撞检测、奖励计算、回合重置、日志统计以及渲染调度。
- `gym_art/quadrotor_multi/quadrotor_single.py`：单架无人机底层环境与观测空间定义，包含自状态、姿态、角速度、房间相关观测等。
- `gym_art/quadrotor_multi/quadrotor_dynamics.py`、`quadrotor_control.py`、`sensor_noise.py`：动力学、控制和传感噪声实现，是论文“直接输出四旋翼推力”的物理基础。
- `gym_art/quadrotor_multi/aerodynamics/downwash.py`、`quad_utils.py`：分别补充多机下洗气动耦合，以及环境主链共享的旋转/观测维度/批量几何工具。
- `gym_art/quadrotor_multi/scenarios/base.py`、`scenarios/utils.py`：把高层任务模式翻译成具体编队类型、层间距和三维目标点，是 episode 几何结构的入口。
- `gym_art/quadrotor_multi/scenarios/mix.py` 与若干 `scenarios/*same_goal.py`：负责在多任务训练里抽取真实 scenario，并实现最基础的静态/动态同目标编队任务。
- `gym_art/quadrotor_multi/quadrotor_randomization.py`：负责 domain randomization 下动力学参数的采样、重采样与物理范围约束。
- `swarm_rl/env_wrappers/reward_shaping.py`、`swarm_rl/env_wrappers/quad_utils.py`：把奖励系数、退火策略、环境包装和 Sample Factory 接口接起来；其中 `quadcol_bin`、`quadcol_bin_smooth_max`、`quadcol_bin_obst` 分别对应机间碰撞、平滑接近惩罚和障碍碰撞惩罚。

### 6.3 论文中的 SDF 障碍观测

- `gym_art/quadrotor_multi/obstacles/obstacles.py`：`MultiObstacles` 在 reset/step 时给每架无人机拼接 9 维障碍观测。
- `gym_art/quadrotor_multi/obstacles/utils.py`：`get_surround_sdfs` 用 `3x3` 局部网格和 `0.1m` 分辨率计算最近障碍距离，正对应论文里描述的 9 维、数量与排列无关的 SDF 风格障碍观测。
- `gym_art/quadrotor_multi/collisions/obstacles.py`：障碍碰撞后的法向与速度更新逻辑。

### 6.4 论文中的注意力模型

- `swarm_rl/models/quad_multi_model.py`：核心网络定义。
- `QuadMultiHeadAttentionEncoder`：当 `quads_encoder_type=attention` 时启用。它先分别编码 self / neighbor / obstacle 三类观测，再把 neighbor 和 obstacle embedding 拼成长度为 2 的 token 序列，送入多头注意力，再与 self embedding 融合输出。
- `QuadMultiEncoder`：非 attention 分支，对应论文早期/对照结构，使用 MLP 或其他邻居编码器。
- `swarm_rl/models/attention_layer.py`：多头注意力与单头注意力算子；其中单头版本用于 `sim2real` 小模型部署。

### 6.5 论文中的回放机制

- `gym_art/quadrotor_multi/quad_experience_replay.py`：论文回放机制的直接实现。
- 核心思路是：每隔一段时间保存环境检查点；一旦发生独特碰撞，就把“碰撞前约 1.5 秒”的状态存进 replay buffer；新 episode 开始时按概率采样这些历史碰撞片段重新训练。
- 这与论文中“放大碰撞事件、从裁剪后的碰撞片段继续训练”的描述一致，也是障碍密集环境下稳定训练的重要部分。

### 6.6 训练后分析、评估与复现实验辅助

- `swarm_rl/enjoy.py`：加载训练好的模型进行可视化评估。
- `swarm_rl/env_wrappers/v_value_map.py`、`paper/plot_v_value_*`：用于论文里的 V-value 可视化分析。
- `paper/*.py`：生成论文中的统计图、对比图和热力图。
- `swarm_rl/sim2real/`：面向部署的小模型与 C/torch 导出、测试逻辑，对应论文后半部分的板载部署讨论。

## 7. 当前人工注释进度补充

- 已完成的环境基础补充文件还包括 `annotated_python/gym_art/quadrotor_multi/aerodynamics/downwash.py`、`annotated_python/gym_art/quadrotor_multi/quad_utils.py`，以及一批场景层文件：`base.py`、`utils.py`、`mix.py`、`dynamic_same_goal.py`、`static_same_goal.py`、`swap_goals.py`、`run_away.py`、`swarm_vs_swarm.py`、`dynamic_diff_goal.py`、`dynamic_formations.py`、`static_diff_goal.py`、`ep_rand_bezier.py`、`ep_lissajous3D.py`、`__init__.py`、`obstacles/o_base.py`、`obstacles/o_random.py`、`test/o_test.py`、`obstacles/o_static_same_goal.py`、`obstacles/o_dynamic_same_goal.py`、`obstacles/o_swap_goals.py`、`obstacles/o_ep_rand_bezier.py`、`test/__init__.py`，以及 `paper/attn_heatmap.py`、`paper/fps_compare.py`、`paper/mean_std_plots_quad_obstacle.py`。
- 这意味着主链已经从训练入口一路覆盖到环境、场景、障碍任务和一部分论文分析图脚本。
- 下一批更适合继续沿 `paper/mean_std_plots_*` 系列推进，把主要实验汇总图脚本补齐。
