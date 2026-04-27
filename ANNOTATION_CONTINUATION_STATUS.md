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
- [annotated_python/swarm_rl/env_wrappers/v_value_map.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/env_wrappers/v_value_map.py:1)
- [annotated_python/gym_art/quadrotor_multi/tests/plot_v_value.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/tests/plot_v_value.py:1)
- [annotated_python/gym_art/quadrotor_multi/tests/plot_v_value_2d.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/tests/plot_v_value_2d.py:1)
- [annotated_python/gym_art/quadrotor_multi/tests/plot_v_value_3d.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/tests/plot_v_value_3d.py:1)
- [annotated_python/gym_art/quadrotor_multi/tests/plot_v_value_4d.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/tests/plot_v_value_4d.py:1)
- [annotated_python/swarm_rl/enjoy.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/enjoy.py:1)
- [annotated_python/swarm_rl/models/weight_recycler.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/models/weight_recycler.py:1)
- [annotated_python/swarm_rl/utils.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/utils.py:1)
- [annotated_python/gym_art/quadrotor_multi/inertia.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/inertia.py:1)
- [annotated_python/gym_art/quadrotor_multi/numba_utils.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/numba_utils.py:1)
- [annotated_python/gym_art/quadrotor_multi/quad_models.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/quad_models.py:1)
- [annotated_python/swarm_rl/env_wrappers/compatibility.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/env_wrappers/compatibility.py:1)
- [annotated_python/swarm_rl/env_wrappers/tests/test_quads.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/swarm_rl/env_wrappers/tests/test_quads.py:1)
- [annotated_python/gym_art/quadrotor_multi/tests/test_numba_opt.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/tests/test_numba_opt.py:1)
- [annotated_python/gym_art/quadrotor_multi/tests/test_multi_env.py](/home/server2/sui_work_not_delete/quad-swarm-rl/annotated_python/gym_art/quadrotor_multi/tests/test_multi_env.py:1)

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
- `paper/mean_std_plots_quad_baseline.py` 中无障碍 baseline 实验怎样把 reward / distance / collision / flight performance 四项指标统一换算成论文图
- `paper/mean_std_plots_quad_annealing.py` 中 baseline / no_annealing / no_replay 三组训练设置怎样共用同一套插值与对数坐标逻辑做并排比较
- `paper/mean_std_plots_quad_obstacle_ablation.py` 中四个障碍实验消融组怎样在 EMA + 平滑后统一比较成功率、碰撞率和目标距离
- `paper/mean_std_plots_quad_scale.py` 中不同 swarm 规模主实验怎样把多组 run 汇总成目标距离与机间碰撞率对比图
- `paper/mean_std_plots_quad_obstacle_num_agents.py` 中 8 / 16 / 32 agent 障碍实验怎样共用 EMA 与平滑管线比较四类主指标
- `swarm_rl/runs/obstacles/quad_obstacle_baseline.py` 中论文障碍主实验的基础 CLI 怎样把回放、退火、障碍观测和默认地图统计打包成复用模板
- `swarm_rl/runs/obstacles/quads_multi_obstacles.py` 中最终 attention 障碍实验怎样在 baseline 之上只覆写邻居观测与 WandB 分组
- `swarm_rl/runs/obstacles/quads_multi_obstacles_nei_encoder_search.py` 中 neighbor encoder 搜索怎样联合扫描可见邻居数和编码器类型
- `swarm_rl/runs/obstacles/pbt_quads_multi_obstacles.py` 中 PBT 版本怎样改写训练时长、worker 布局和策略进化参数
- `swarm_rl/runs/obstacles/obst_density_random.py`、`obst_domain_random.py`、`obst_size_random.py` 中三类 obstacle randomization 配置怎样分别控制 density、size 或二者同时随机化
- `swarm_rl/runs/single_quad/baseline.py` 中单机训练模板怎样把多机环境退化成 `quads_num_agents=1` 的基础飞行配置
- `swarm_rl/runs/single_quad/single_quad.py` 中单机实验入口怎样在 baseline 之上只展开多 seed 和本地验证用的 WandB 标签
- `swarm_rl/env_wrappers/v_value_map.py` 中 wrapper 怎样围绕当前观测做局部 x/y 扰动，并把 critic 的 value map 作为第二张图拼接到渲染帧旁边
- `gym_art/quadrotor_multi/tests/plot_v_value_2d.py` 中运行期真正被调用的 helper 怎样把 21x21 value 栅格变成 RGB 热图
- `plot_v_value.py`、`plot_v_value_3d.py`、`plot_v_value_4d.py` 怎样作为离线 debugger 工具，从人工导出的 value 样本检查 critic 在 1D/2D/3D 切片上的形状
- `swarm_rl/enjoy.py` 中评估入口怎样复用训练期的组件注册与配置解析，并切到 Sample Factory 的 checkpoint rollout 入口
- `swarm_rl/models/weight_recycler.py` 中逐神经元平均绝对激活分数怎样被提炼出来，供上层判断通道是否长期沉寂
- `swarm_rl/utils.py` 中时间戳命名与多 seed 生成这两个小工具怎样服务实验脚本的批量运行和产物管理
- `inertia.py` 中完整机体怎样被拆成多个刚体部件，并通过姿态旋转、质心重算与平行轴定理装配成整机惯量
- `numba_utils.py` 中动力学热点所需的 `clip` overload、推力映射、叉乘和 OU 噪声怎样被做成 numba 友好的实现
- `quad_models.py` 中几套标准机体模板怎样统一打包几何、阻尼、噪声和电机参数
- `quadrotor_visualization.py` 中单机动力学状态、goal、追踪/侧视相机和 scene graph 怎样被串成第三人称回放与第一人称观测渲染链
- `quadrotor_multi_visualization.py` 中多机 goal、障碍物、碰撞球、路径尾迹、速度/加速度箭头与键盘切视角逻辑怎样统一调度
- `rendering3d.py` 中 window/FBO target、camera、scene graph、程序纹理与基础几何 primitive 怎样组成底层 OpenGL 渲染框架
- `swarm_rl/env_wrappers/compatibility.py` 中旧式 `(obs, reward, done, info)` 多机环境怎样被兼容层改写成 Gymnasium / Sample Factory 期望的 terminated-truncated API
- `swarm_rl/env_wrappers/tests/test_quads.py` 中组件注册、训练配置解析、环境创建与多步 rollout 这条 Sample Factory 入口链怎样被做成冒烟测试
- `gym_art/quadrotor_multi/tests/test_numba_opt.py` 中环境 step 吞吐、动力学 `step1` 和传感器噪声路径怎样对拍 python 实现与 numba 实现
- `gym_art/quadrotor_multi/tests/test_multi_env.py` 中基础多机环境、渲染、本地观测和 replay wrapper 几条常用测试链怎样被统一验证

## 5. 还未处理的重点文件

`paper/mean_std_plots_*` 主脚本、`swarm_rl/runs/obstacles/` 配置链、`swarm_rl/runs/single_quad/` 配置链、一组 `V-value` 可视化脚本，以及单机/多机/底层 3D 渲染链已经收尾，下一批建议转去零散尾项：

1. 剩余测试、可视化零散文件
2. 其它尚未人工重写的辅助工具文件

原因：

- `paper/mean_std_plots_*` 这条连续主线已经全部补齐，继续停留在这一组文件上收益不高
- `swarm_rl/runs/obstacles/` 这条和论文障碍复现实验最贴近的配置链也已经补到较完整状态
- `swarm_rl/runs/single_quad/` 这条单机配置链也已经收尾
- `V-value` 可视化链里运行期 wrapper 与离线 debugger 脚本也已经说明清楚
- 下一步更适合把剩余测试脚本、零散可视化或还没人工重写的小工具文件收尾，减少断点分散

之后建议顺序：

1. 剩余测试、可视化零散文件
2. 其它尚未人工重写的辅助工具文件

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
   `mean_std_plots_quad_obstacle_compare_arch_neighbor.py` -> `mean_std_plots_quad_baseline.py` ->
   `mean_std_plots_quad_annealing.py` -> `mean_std_plots_quad_obstacle_ablation.py` ->
   `mean_std_plots_quad_scale.py` -> `mean_std_plots_quad_obstacle_num_agents.py`
10. V-value 可视化尾项：
   `swarm_rl/env_wrappers/v_value_map.py` -> `tests/plot_v_value.py` ->
   `tests/plot_v_value_2d.py` -> `tests/plot_v_value_3d.py` -> `tests/plot_v_value_4d.py`
11. 3D 渲染显示链：
   `quadrotor_visualization.py` -> `quadrotor_multi_visualization.py` -> `rendering3d.py`

这个顺序的核心原则是：

- 先把“训练配置 -> 环境主循环 -> 观测/奖励/碰撞 -> 模型”打通
- 再去注释场景文件，否则很难解释 `goal`、`spawn_points`、`distance_to_goal` 是怎样被外层环境消费的
- 最后回到 `paper/`，这样才能把作图脚本和前面的训练日志、统计量、场景指标链路接起来

## 5.2 下一次继续时的推荐顺序

如果下一次继续，建议不要跳着做，而是按下面顺序推进剩余零散文件：

1. `gym_art/quadrotor_multi/obstacles/test/unit_test.py`
2. `gym_art/quadrotor_multi/obstacles/test/speed_test.py`
3. `gym_art/quadrotor_multi/collisions/test/unit_test/obstacles.py`
4. `gym_art/quadrotor_multi/collisions/test/unit_test/quadrotor.py`
5. 其余剩余测试、可视化零散文件

推荐理由：

- 环境包装层与基础测试链已经补齐，下一步最顺的是把障碍物与碰撞相关的单元/速度测试一起收尾
- `obstacles/test/*` 紧挨着前面已做过的 `obstacles.py` 与 `obstacles/utils.py`，上下文最连续
- `collisions/test/unit_test/*` 则正好接在已做过的 `collisions/obstacles.py` 与 `collisions/quadrotors.py` 后面，能把验证链补完整

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
