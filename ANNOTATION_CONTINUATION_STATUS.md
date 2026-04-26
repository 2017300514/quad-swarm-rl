# 注释续接状态说明

## 1. 任务目标

当前任务不是修改源码，而是**只修改 `annotated_python/` 中的注释副本**，把原先偏模板化、偏语法翻译式的注释，改成**贴合项目实际逻辑的工程语义注释**。

目标注释应当解释：

- 数据从哪里来
- 为什么在这里被修改
- 修改后会被谁继续使用
- 这段逻辑在训练、仿真、碰撞、观测、奖励、回放、部署中的作用

明确不要做的事：

- 不修改原始源码目录 `gym_art/`、`swarm_rl/`、`paper/`
- 不再用脚本批量生成模板注释
- 不写“保存或更新变量”“调用某函数执行当前处理”这类空泛注释

## 2. 已确定的工作原则

本轮已经和用户确认，后续必须按**人工逐文件阅读、人工逐文件落注**的流程继续。

正式流程已经写入：

- [ANNOTATION_AND_PAPER_GUIDE.md](/home/server2/sui_work_not_delete/quad-swarm-rl/ANNOTATION_AND_PAPER_GUIDE.md:1)

这份文档已经改成当前有效版本，核心约束如下：

1. 先读原始源码，再改 `annotated_python/` 副本。
2. 注释要解释模块职责、上下游数据流、关键状态、关键代码块和论文机制。
3. 优先处理主链路，再处理外围模块。
4. 每完成一批，要同步更新工作记录。

## 3. 已处理文件

以下文件已经按新的人工方式重写，不再是旧模板风格：

- [annotated_python/swarm_rl/train.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/train.py:1)
- [annotated_python/swarm_rl/env_wrappers/quadrotor_params.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/env_wrappers/quadrotor_params.py:1)
- [annotated_python/swarm_rl/env_wrappers/quad_utils.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/env_wrappers/quad_utils.py:1)
- [annotated_python/swarm_rl/env_wrappers/reward_shaping.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/env_wrappers/reward_shaping.py:1)
- [annotated_python/gym_art/quadrotor_multi/quadrotor_single.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/quadrotor_single.py:1)
- [annotated_python/gym_art/quadrotor_multi/quadrotor_multi.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/quadrotor_multi.py:1)
- [annotated_python/gym_art/quadrotor_multi/quadrotor_dynamics.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/quadrotor_dynamics.py:1)
- [annotated_python/gym_art/quadrotor_multi/quadrotor_control.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/quadrotor_control.py:1)
- [annotated_python/gym_art/quadrotor_multi/obstacles/obstacles.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/obstacles/obstacles.py:1)
- [annotated_python/gym_art/quadrotor_multi/obstacles/utils.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/obstacles/utils.py:1)
- [annotated_python/gym_art/quadrotor_multi/collisions/obstacles.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/collisions/obstacles.py:1)
- [annotated_python/gym_art/quadrotor_multi/collisions/quadrotors.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/collisions/quadrotors.py:1)
- [annotated_python/gym_art/quadrotor_multi/quad_experience_replay.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/quad_experience_replay.py:1)
- [annotated_python/swarm_rl/models/quad_multi_model.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/models/quad_multi_model.py:1)
- [annotated_python/swarm_rl/models/attention_layer.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/models/attention_layer.py:1)
- [annotated_python/gym_art/quadrotor_multi/sensor_noise.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/sensor_noise.py:1)
- [annotated_python/gym_art/quadrotor_multi/collisions/utils.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/collisions/utils.py:1)
- [annotated_python/swarm_rl/sim2real/code_blocks.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/sim2real/code_blocks.py:1)
- [annotated_python/swarm_rl/sim2real/sim2real.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/sim2real/sim2real.py:1)
- [annotated_python/swarm_rl/sim2real/torch_models/__init__.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/sim2real/torch_models/__init__.py:1)
- [annotated_python/swarm_rl/sim2real/c_models/__init__.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/sim2real/c_models/__init__.py:1)
- [annotated_python/swarm_rl/sim2real/tests/unit_tests.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/sim2real/tests/unit_tests.py:1)
- [annotated_python/gym_art/quadrotor_multi/get_state.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/get_state.py:1)
- [annotated_python/gym_art/quadrotor_multi/collisions/room.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/collisions/room.py:1)
- [annotated_python/gym_art/quadrotor_multi/aerodynamics/downwash.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/aerodynamics/downwash.py:1)
- [annotated_python/gym_art/quadrotor_multi/quad_utils.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/quad_utils.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/base.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/base.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/utils.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/utils.py:1)
- [annotated_python/gym_art/quadrotor_multi/quadrotor_randomization.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/quadrotor_randomization.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/mix.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/mix.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_same_goal.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_same_goal.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/static_same_goal.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/static_same_goal.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/swap_goals.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/swap_goals.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/run_away.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/run_away.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/swarm_vs_swarm.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/swarm_vs_swarm.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_diff_goal.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_diff_goal.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_formations.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_formations.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/static_diff_goal.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/static_diff_goal.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/ep_rand_bezier.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/ep_rand_bezier.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/ep_lissajous3D.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/ep_lissajous3D.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/__init__.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/__init__.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_base.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_base.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_random.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_random.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/test/o_test.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/test/o_test.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_static_same_goal.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_static_same_goal.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_dynamic_same_goal.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_dynamic_same_goal.py:1)
- [annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_swap_goals.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_swap_goals.py:1)

## 4. 已完成内容概述

### 4.1 训练入口链路

已完成：

- `train.py` 中 Sample Factory 注册、两阶段参数解析、训练主循环接线说明
- `quadrotor_params.py` 中各类参数如何流向环境、观测、奖励、障碍物、回放和 sim2real
- `quad_utils.py` 中配置如何实例化为环境对象与 wrapper 链
- `reward_shaping.py` 中 true reward、episode 级统计、碰撞惩罚退火逻辑

### 4.2 环境主链路

已完成：

- `quadrotor_single.py` 中单机物理单元、基础奖励、观测空间、动力学采样、reset/step
- `quadrotor_multi.py` 中多机聚合、邻居选择、障碍物生成、碰撞处理、额外奖励、场景推进、episode 指标
- `quadrotor_dynamics.py` 中电机滞后、推力/力矩计算、姿态积分、墙面/天花板/地面接触、numba 加速路径
- `quadrotor_control.py` 中 raw control、简化控制接口、雅可比反解、速度/位置解析控制到四电机推力的映射
- `obstacles/obstacles.py` 中障碍物布局缓存、9 维障碍观测拼接和无人机-障碍索引碰撞输出
- `obstacles/utils.py` 中 3x3 局部 SDF 计算、障碍碰撞编号和障碍生成网格中心枚举
- `collisions/obstacles.py` 中障碍碰撞法向、弹开后的速度修正和角速度扰动
- `collisions/quadrotors.py` 中机间碰撞对扫描、接近惩罚和碰撞后双方速度/角速度修正
- `quad_experience_replay.py` 中检查点保存、碰撞前 1.5 秒片段裁剪、回放采样和多机自动 reset 接管
- `quad_multi_model.py` 中 self / neighbor / obstacle 观测切片、邻居聚合、注意力编码和 sim2real 轻量分支
- `attention_layer.py` 中多头/单头注意力和最底层 scaled dot-product attention 的张量流
- `sensor_noise.py` 中位置/速度/姿态/陀螺仪/加速度计噪声如何从理想状态生成观测扰动
- `collisions/utils.py` 中碰撞后速度衰减和角速度扰动的底层公用数值逻辑
- `sim2real/code_blocks.py` 中板载控制器源码模板、注意力 C 展开实现和 state vector 到 PWM 的导出片段
- `sim2real/sim2real.py` 中训练模型重建、权重转 C 字符串、single/attention 两条导出流程
- `sim2real/torch_models/__init__.py` 中 torch 模型目录作为部署资源包入口的角色
- `sim2real/c_models/__init__.py` 中 C 模型产物目录作为部署资源包入口的角色
- `sim2real/tests/unit_tests.py` 中 PyTorch 与 C 导出模型的逐层/最终输出一致性对拍流程
- `get_state.py` 中带噪动力学状态如何整理成不同 `obs_repr` 的自观测向量
- `collisions/room.py` 中墙面/天花板碰撞后的速度与角速度修正
- `aerodynamics/downwash.py` 中多机垂直叠放时的简化下洗流扰动、作用范围和速度/角速度注入方式
- `quad_utils.py` 中环境与模型共享的观测维度约定、旋转几何工具和批量叉乘等基础数值辅助逻辑
- `scenarios/base.py` 中任务模式怎样生成编队 goals，以及 formation size / layer distance 怎样在 reset 中被采样
- `scenarios/utils.py` 中场景模式枚举、编队尺寸换算、球面/圆环/网格目标几何辅助函数
- `quadrotor_randomization.py` 中 domain randomization 如何约束、扰动或重采样机体动力学参数
- `scenarios/mix.py` 中混合任务模式怎样在每个 episode 随机挑选真实 scenario，并把其输出同步到外层环境
- `scenarios/dynamic_same_goal.py` 中同目标编队怎样按时间间隔整体平移到新中心
- `scenarios/static_same_goal.py` 中最简单的静态同目标基线场景
- `scenarios/swap_goals.py` 中固定目标集合怎样只交换 agent-goal 映射，制造周期性任务重分配
- `scenarios/run_away.py` 中前两架无人机怎样周期性改追其余 agent 的目标，制造局部追逃扰动
- `scenarios/swarm_vs_swarm.py` 中两支子群怎样分别成队、周期性交换目标中心并触发双群交汇
- `scenarios/dynamic_diff_goal.py` 中整组不同目标怎样周期性重采中心、编队几何和分配关系
- `scenarios/dynamic_formations.py` 中编队尺度怎样在 episode 中连续伸缩并持续改写相对目标
- `scenarios/static_diff_goal.py` 中最基础的不同目标静态基线怎样作为动态重分配任务的对照组
- `scenarios/ep_rand_bezier.py` 中共享移动目标怎样按 5 秒时间窗重采 Bezier 轨迹
- `scenarios/ep_lissajous3D.py` 中共享目标怎样沿固定三维 Lissajous 周期轨迹运动
- `scenarios/__init__.py` 中空目录入口文件在包导入链上的角色
- `scenarios/obstacles/o_base.py` 中障碍地图怎样被转成自由栅格采样、spawn_points 和逐段目标切换逻辑
- `scenarios/obstacles/o_random.py` 中最常见障碍随机任务怎样为每架无人机直接采样独立起终点
- `scenarios/test/o_test.py` 中测试场景怎样用固定两侧起终点快速构造穿越任务
- `scenarios/obstacles/o_static_same_goal.py` 中障碍同目标静态基线怎样让所有无人机汇聚到同一个安全终点
- `scenarios/obstacles/o_dynamic_same_goal.py` 中障碍动态同目标任务怎样限制共享目标的跳变距离
- `scenarios/obstacles/o_swap_goals.py` 中障碍目标交换任务怎样在安全目标槽位内部重分配 agent-goal 映射
- `scenarios/obstacles/o_ep_rand_bezier.py` 中障碍环境里的共享移动目标怎样沿 Bezier 轨迹平滑推进
- `scenarios/test/__init__.py` 中测试子目录空入口文件在导入链上的角色
- `paper/attn_heatmap.py` 中论文注意力热力图的矩阵来源与对比意图
- `paper/fps_compare.py` 中仿真吞吐量柱状图怎样对比 QuadSwarm 与 gym-pybullet-drones
- `paper/mean_std_plots_quad_obstacle.py` 中障碍实验 TensorBoard 日志怎样被聚合成四联图
- `paper/mean_std_plots_quad_compare_arch.py` 中 attention / deepsets / mlp 三组架构日志怎样被聚合成统一对比图
- `paper/mean_std_plots_quad_obstacle_compare_arch_density.py` 中不同 obstacle density 分组怎样影响成功率、碰撞率和目标距离曲线
- `paper/mean_std_plots_quad_obstacle_compare_arch_neighbor.py` 中不同 neighbor budget 分组怎样影响障碍实验的四类主指标

## 5. 还未处理的重点文件

下一批应优先继续下面三个文件：

1. `annotated_python/paper/mean_std_plots_quad_baseline.py`
2. `annotated_python/paper/mean_std_plots_quad_annealing.py`
3. `annotated_python/paper/mean_std_plots_quad_obstacle_ablation.py`

原因：

- 当前 `paper/` 目录已经补到 compare 系列，继续顺着剩余 `mean_std_plots_*` 聚合脚本推进，最容易保持注释口径一致
- `mean_std_plots_quad_baseline.py` 与 `mean_std_plots_quad_annealing.py` 承接无障碍主实验与奖励/退火分析
- `mean_std_plots_quad_obstacle_ablation.py` 承接障碍实验中的组件消融对比，和刚完成的 density / neighbor 图逻辑最接近

之后建议顺序：

1. `annotated_python/paper/mean_std_plots_quad_baseline.py`
2. `annotated_python/paper/mean_std_plots_quad_annealing.py`
3. `annotated_python/paper/mean_std_plots_quad_obstacle_ablation.py`
4. `annotated_python/paper/mean_std_plots_quad_scale.py`
5. `annotated_python/paper/mean_std_plots_quad_obstacle_num_agents.py`
6. 剩余测试、可视化零散文件

## 5.1 本轮实际注释顺序摘要

为了保证前后注释口径一致，本轮人工重写实际遵循的是“主链先打通，再沿分支往外扩，再回到论文脚本”的顺序，而不是单纯按目录字母序：

1. 训练入口与环境包装：
   `train.py` -> `quadrotor_params.py` -> `quad_utils.py` -> `reward_shaping.py`
2. 多机环境主链：
   `quadrotor_single.py` -> `quadrotor_multi.py` -> `quadrotor_dynamics.py` -> `quadrotor_control.py`
3. 障碍物、碰撞与回放支链：
   `obstacles.py` -> `obstacles/utils.py` -> `collisions/obstacles.py` -> `collisions/quadrotors.py` -> `quad_experience_replay.py`
4. 模型与部署支链：
   `quad_multi_model.py` -> `attention_layer.py` -> `sensor_noise.py` -> `collisions/utils.py` ->
   `sim2real/code_blocks.py` -> `sim2real/sim2real.py` -> `sim2real/torch_models/__init__.py` ->
   `sim2real/c_models/__init__.py` -> `sim2real/tests/unit_tests.py` -> `get_state.py` ->
   `collisions/room.py` -> `aerodynamics/downwash.py` -> `quad_utils.py`
5. 场景主链：
   `scenarios/base.py` -> `scenarios/utils.py` -> `quadrotor_randomization.py` -> `scenarios/mix.py`
6. 普通场景变体：
   `dynamic_same_goal.py` -> `static_same_goal.py` -> `swap_goals.py` -> `run_away.py` ->
   `swarm_vs_swarm.py` -> `dynamic_diff_goal.py` -> `dynamic_formations.py` -> `static_diff_goal.py`
7. 轨迹场景与入口：
   `ep_rand_bezier.py` -> `ep_lissajous3D.py` -> `scenarios/__init__.py`
8. 障碍场景分支：
   `obstacles/o_base.py` -> `obstacles/o_random.py` -> `test/o_test.py` ->
   `obstacles/o_static_same_goal.py` -> `obstacles/o_dynamic_same_goal.py` ->
   `obstacles/o_swap_goals.py` -> `obstacles/o_ep_rand_bezier.py` -> `test/__init__.py`
9. 论文分析脚本：
   `attn_heatmap.py` -> `fps_compare.py` -> `mean_std_plots_quad_obstacle.py` ->
   `mean_std_plots_quad_compare_arch.py` -> `mean_std_plots_quad_obstacle_compare_arch_density.py` ->
   `mean_std_plots_quad_obstacle_compare_arch_neighbor.py`

这个顺序的核心原则是：

- 先把“训练配置 -> 环境主循环 -> 观测/奖励/碰撞 -> 模型”打通
- 再去注释场景文件，否则很难解释 `goal`、`spawn_points`、`distance_to_goal` 是怎样被外层环境消费的
- 最后回到 `paper/`，这样才能把作图脚本和前面的训练日志、统计量、场景指标链路接起来

## 5.2 下一次继续时的推荐顺序

如果下一次继续，建议不要跳着做，而是按下面顺序推进剩余 `paper/` 文件：

1. `annotated_python/paper/mean_std_plots_quad_baseline.py`
2. `annotated_python/paper/mean_std_plots_quad_annealing.py`
3. `annotated_python/paper/mean_std_plots_quad_obstacle_ablation.py`
4. `annotated_python/paper/mean_std_plots_quad_scale.py`
5. `annotated_python/paper/mean_std_plots_quad_obstacle_num_agents.py`

推荐理由：

- 这几份脚本和刚完成的 compare 系列最相邻，数据流、缓存、插值、平滑、均值/标准差带的逻辑最容易复用
- `baseline` 与 `annealing` 先做，可以把无障碍主实验和奖励退火逻辑补完整
- `obstacle_ablation`、`scale`、`num_agents` 再跟进，可以把障碍实验横向对比链补齐
- 这样收完以后，`paper/` 目录只会剩少量零散可视化脚本，断点最清楚

## 6. 当前有效注释风格

后续注释必须延续下面这种风格：

- 文件头先说明模块职责、上游输入、下游消费者
- 关键函数说明“它完成哪段流程”，不是“定义了什么函数”
- 关键变量说明来源、用途、后续流向
- 关键代码块解释训练/仿真含义，而不是逐行翻译
- 尽量按“主线块注释 + 局部关键变量解释”的方式组织

禁止恢复成下面这些写法：

- “导入当前模块依赖”
- “定义函数/类”
- “保存或更新变量”
- “调用某函数执行当前处理”
- “根据条件决定是否进入分支”

## 7. 已做过的检查

本轮至少做过这些检查：

- 新改写的 `annotated_python/gym_art/quadrotor_multi/quadrotor_single.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/quadrotor_multi.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/quadrotor_dynamics.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/quadrotor_control.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/obstacles/obstacles.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/obstacles/utils.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/collisions/obstacles.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/collisions/quadrotors.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/quad_experience_replay.py`
- 新改写的 `annotated_python/swarm_rl/models/quad_multi_model.py`
- 新改写的 `annotated_python/swarm_rl/models/attention_layer.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/sensor_noise.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/collisions/utils.py`
- 新改写的 `annotated_python/swarm_rl/sim2real/code_blocks.py`
- 新改写的 `annotated_python/swarm_rl/sim2real/sim2real.py`
- 新改写的 `annotated_python/swarm_rl/sim2real/torch_models/__init__.py`
- 新改写的 `annotated_python/swarm_rl/sim2real/c_models/__init__.py`
- 新改写的 `annotated_python/swarm_rl/sim2real/tests/unit_tests.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/get_state.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/collisions/room.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/aerodynamics/downwash.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/quad_utils.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/base.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/utils.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/quadrotor_randomization.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/mix.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_same_goal.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/static_same_goal.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/swap_goals.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/run_away.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/swarm_vs_swarm.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_diff_goal.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_formations.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/static_diff_goal.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/ep_rand_bezier.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/ep_lissajous3D.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/__init__.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_base.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_random.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/test/o_test.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_static_same_goal.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_dynamic_same_goal.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_swap_goals.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_ep_rand_bezier.py`
- 新改写的 `annotated_python/gym_art/quadrotor_multi/scenarios/test/__init__.py`
- 新改写的 `annotated_python/paper/attn_heatmap.py`
- 新改写的 `annotated_python/paper/fps_compare.py`
- 新改写的 `annotated_python/paper/mean_std_plots_quad_obstacle.py`

已通过：

```bash
python -m py_compile annotated_python/gym_art/quadrotor_multi/quadrotor_single.py annotated_python/gym_art/quadrotor_multi/quadrotor_multi.py annotated_python/gym_art/quadrotor_multi/quadrotor_dynamics.py annotated_python/gym_art/quadrotor_multi/quadrotor_control.py annotated_python/gym_art/quadrotor_multi/obstacles/obstacles.py annotated_python/gym_art/quadrotor_multi/obstacles/utils.py annotated_python/gym_art/quadrotor_multi/collisions/obstacles.py annotated_python/gym_art/quadrotor_multi/collisions/quadrotors.py annotated_python/gym_art/quadrotor_multi/quad_experience_replay.py annotated_python/swarm_rl/models/quad_multi_model.py annotated_python/swarm_rl/models/attention_layer.py annotated_python/gym_art/quadrotor_multi/sensor_noise.py annotated_python/gym_art/quadrotor_multi/collisions/utils.py annotated_python/swarm_rl/sim2real/code_blocks.py annotated_python/swarm_rl/sim2real/sim2real.py annotated_python/swarm_rl/sim2real/torch_models/__init__.py annotated_python/swarm_rl/sim2real/c_models/__init__.py annotated_python/swarm_rl/sim2real/tests/unit_tests.py annotated_python/gym_art/quadrotor_multi/get_state.py annotated_python/gym_art/quadrotor_multi/collisions/room.py annotated_python/gym_art/quadrotor_multi/aerodynamics/downwash.py annotated_python/gym_art/quadrotor_multi/quad_utils.py annotated_python/gym_art/quadrotor_multi/scenarios/base.py annotated_python/gym_art/quadrotor_multi/scenarios/utils.py annotated_python/gym_art/quadrotor_multi/quadrotor_randomization.py annotated_python/gym_art/quadrotor_multi/scenarios/mix.py annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_same_goal.py annotated_python/gym_art/quadrotor_multi/scenarios/static_same_goal.py annotated_python/gym_art/quadrotor_multi/scenarios/swap_goals.py annotated_python/gym_art/quadrotor_multi/scenarios/run_away.py annotated_python/gym_art/quadrotor_multi/scenarios/swarm_vs_swarm.py annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_diff_goal.py annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_formations.py annotated_python/gym_art/quadrotor_multi/scenarios/static_diff_goal.py annotated_python/gym_art/quadrotor_multi/scenarios/ep_rand_bezier.py annotated_python/gym_art/quadrotor_multi/scenarios/ep_lissajous3D.py annotated_python/gym_art/quadrotor_multi/scenarios/__init__.py annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_base.py annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_random.py annotated_python/gym_art/quadrotor_multi/scenarios/test/o_test.py annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_static_same_goal.py annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_dynamic_same_goal.py annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_swap_goals.py annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_ep_rand_bezier.py annotated_python/gym_art/quadrotor_multi/scenarios/test/__init__.py
```

并且已检查过这些文件中没有残留旧模板短语。

## 8. 关键注意事项

1. `annotated_python/` 里还有大量旧模板注释文件未重写，不要把“已存在注释副本”误认为“已完成人工注释”。
2. 旧脚本 [tools/generate_chinese_annotations.py](/home/server2/sui_work_not_delete/quad-swarm-rl/tools/generate_chinese_annotations.py:1) 已标记为 deprecated，不应再用它覆盖副本。
3. 后续每处理完一批文件，都要同步更新 [ANNOTATION_AND_PAPER_GUIDE.md](/home/server2/sui_work_not_delete/quad-swarm-rl/ANNOTATION_AND_PAPER_GUIDE.md:1)。
4. 如果继续做大文件，优先做“块级高价值注释”，不要追求逐行满注释。

## 9. 下次续接时建议直接执行的动作

下次继续时，建议先做下面几步：

1. 先阅读：
   - [ANNOTATION_CONTINUATION_STATUS.md](/home/server2/sui_work_not_delete/quad-swarm-rl/ANNOTATION_CONTINUATION_STATUS.md:1)
   - [ANNOTATION_AND_PAPER_GUIDE.md](/home/server2/sui_work_not_delete/quad-swarm-rl/ANNOTATION_AND_PAPER_GUIDE.md:1)
2. 继续阅读原始源码：
   - `paper/attn_heatmap.py`
   - `paper/fps_compare.py`
   - `paper/mean_std_plots_quad_obstacle.py`
3. 只改对应注释副本：
   - `annotated_python/paper/attn_heatmap.py`
   - `annotated_python/paper/fps_compare.py`
   - `annotated_python/paper/mean_std_plots_quad_obstacle.py`
4. 修改完成后更新工作记录。

## 10. 一句话续接指令

如果下次需要快速把任务接起来，可以直接用这段意思继续：

“继续按 `ANNOTATION_CONTINUATION_STATUS.md` 和 `ANNOTATION_AND_PAPER_GUIDE.md` 的人工逐文件流程，从剩余未人工重写的高价值文件继续，只修改 `annotated_python/` 副本，不动源码。”
