# 论文障碍场景复现资产排查与绘图方案

更新时间：`2026-04-26`

## 1. 任务范围

本次排查基于以下材料与代码入口：

- `ANNOTATION_AND_PAPER_GUIDE.md`
- `ANNOTATION_CONTINUATION_STATUS.md`
- `REPRO_HYBRID_AUTO_SETUP.md`
- `hybrid_auto/Huang 等 - Collision Avoidance and Navigation for a Quadrotor.md`

目标论文是：

- `Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning`

本次要回答的问题：

1. 当前工作空间里是否已经有针对论文障碍场景训练好的模型。
2. 是否已有可直接生成论文指标图、轨迹图、动图的脚本与数据链路。
3. `conda activate swarm-rl-obstacles` 当前是否具备这些能力。
4. 是否需要额外平台，例如 ROS / Gazebo / AirSim / sim。
5. 给出一个可执行的绘图与复现实验方案。

## 2. 论文相关主线

从论文和仓库已有说明可以确认，当前最贴近论文障碍实验的代码主线是：

- 训练配置：
  - `swarm_rl/runs/obstacles/quad_obstacle_baseline.py`
  - `swarm_rl/runs/obstacles/quads_multi_obstacles.py`
  - `swarm_rl/runs/obstacles/pbt_quads_multi_obstacles.py`
- 环境与障碍观测：
  - `gym_art/quadrotor_multi/quadrotor_multi.py`
  - `gym_art/quadrotor_multi/obstacles/obstacles.py`
  - `gym_art/quadrotor_multi/obstacles/utils.py`
- 注意力模型：
  - `swarm_rl/models/quad_multi_model.py`
  - `swarm_rl/models/attention_layer.py`
- 回放机制：
  - `gym_art/quadrotor_multi/quad_experience_replay.py`
- 可视化与评估：
  - `swarm_rl/enjoy.py`
  - `swarm_rl/env_wrappers/v_value_map.py`
  - `gym_art/quadrotor_multi/quadrotor_multi_visualization.py`
  - `paper/*.py`

论文里明确提到的障碍实验基线是：

- `8` 架无人机
- `20%` obstacle density
- `0.6m` obstacle size
- `2` 个可感知邻居

这和仓库里的 obstacle run 配置一致。

## 3. 当前工作空间是否已有训练好的论文模型

结论：`没有在当前工作空间内发现可直接使用的论文障碍场景训练权重或训练日志。`

本次已搜索但未发现以下典型产物：

- `checkpoint_*`
- `best_*`
- `cfg.json`
- `events.out.tfevents.*`
- `train_dir/`
- 常见模型文件：`.pth` `.pt` `.ckpt`

因此当前仓库虽然有：

- 论文对应训练配置
- 评估入口
- 出图脚本
- 已有示例 GIF

但`没有找到`能够直接支撑“复现论文数值结果”的本地训练产物。

补充判断：

- `README.md` 里还写到“后续会开源一些重要训练模型”，这和当前未发现预置权重的现状一致。
- 现有 `swarm_rl/gifs/*.gif` 更像仓库展示样例，不是当前工作空间里与某次本地训练目录绑定的可追溯实验输出。

## 4. 已找到的论文相关出图/可视化文件

### 4.1 论文指标曲线脚本

已找到这些脚本：

- `paper/mean_std_plots_quad_obstacle.py`
- `paper/mean_std_plots_quad_obstacle_num_agents.py`
- `paper/mean_std_plots_quad_obstacle_compare_arch_density.py`
- `paper/mean_std_plots_quad_obstacle_compare_arch_neighbor.py`
- `paper/mean_std_plots_quad_obstacle_ablation.py`
- `paper/mean_std_plots_quad_baseline.py`
- `paper/mean_std_plots_quad_annealing.py`
- `paper/mean_std_plots_quad_scale.py`
- `paper/fps_compare.py`
- `paper/attn_heatmap.py`

这些脚本的共同特点：

- 主要读取 `TensorBoard` 的 `*.tfevents.*`。
- 通过 `tensorboard.backend.event_processing.EventAccumulator` 聚合多 seed 曲线。
- 输出一般是 `pdf`。
- 部分脚本会在 `paper/cache/` 下缓存 pickle。
- 部分脚本默认把结果输出到 `paper/../final_plots/` 或 `paper/final_plots/`。

### 4.2 论文指标脚本可读的关键指标

从脚本中已经确认可聚合这些论文相关指标：

- `metric/agent_success_rate`
- `metric/agent_col_rate`
- `metric/agent_obst_col_rate`
- `o_random/distance_to_goal_1s`
- `o_static_same_goal/distance_to_goal_1s`
- 以及若干训练阶段辅助量

这和论文里提到的：

- success rate
- collision rate
- distance to goal
- flight distance
- inference time

是对得上的，但要注意：

- 当前脚本里最直接覆盖的是 success/collision/distance。
- `inference time` 在仓库里更像单独统计或单独出图，例如 `paper/fps_compare.py`，不是从训练日志里自动读出的统一字段。

### 4.3 路径图 / V-value 图 / 渲染路径

已找到：

- `swarm_rl/env_wrappers/v_value_map.py`
- `gym_art/quadrotor_multi/tests/plot_v_value.py`
- `gym_art/quadrotor_multi/tests/plot_v_value_2d.py`
- `gym_art/quadrotor_multi/tests/plot_v_value_3d.py`
- `gym_art/quadrotor_multi/tests/plot_v_value_4d.py`
- `gym_art/quadrotor_multi/plots/plot_v_value_1d.py`
- `gym_art/quadrotor_multi/plots/plot_v_value_2d.py`
- `gym_art/quadrotor_multi/plots/plot_v_value_3d.py`
- `gym_art/quadrotor_multi/plots/plot_v_value_4d.py`

能力判断：

- `V_ValueMapWrapper` 会在 `render(rgb_array)` 时把当前环境帧和 V-value 2D 图拼接。
- 这条链路要求先加载训练好的 checkpoint。
- 所以代码具备“生成论文 Figure 6 风格 V-value map”的能力，但前提仍然是：`先有可加载的模型`。

### 4.4 轨迹渲染 / 动图生成相关代码

已找到：

- `gym_art/quadrotor_multi/quadrotor_multi_visualization.py`
- `gym_art/quadrotor_multi/quadrotor_visualization.py`
- `gym_art/quadrotor_multi/rendering3d.py`
- `swarm_rl/enjoy.py`

能力判断：

- `QuadrotorEnvMulti.render()` 支持 `rgb_array` 返回帧。
- `quadrotor_multi_visualization.py` 内部维护 `path_store`，可以在渲染中显示轨迹点。
- `README.md` 明确给出测试模型的命令：
  - `python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi ... --train_dir=PATH_TO_TRAIN_DIR --experiment=EXPERIMENT_NAME ...`
- `enjoy` 的 CLI 已暴露：
  - `--save_video`
  - `--video_frames`
  - `--video_name`
  - `--max_num_frames`
  - `--max_num_episodes`

因此：

- `mp4`/视频导出主链路优先应走 `sample_factory.enjoy` 的 `--save_video`。
- 仓库本体没有发现一个现成的“把帧批量编码成 GIF”的专用脚本。
- 如果想自己导出 GIF，最稳妥的做法是补 `imageio` 后写一个很短的导出脚本，或先生成视频再转 GIF。

## 5. 当前环境 `swarm-rl-obstacles` 的真实能力

### 5.1 已验证存在的环境

已确认 conda 环境存在：

- `swarm-rl-obstacles`

### 5.2 已验证可用的关键能力

已确认以下包已经安装：

- `torch==2.5.0+cu124`
- `sample-factory==2.1.1`
- `gymnasium==0.28.1`
- `matplotlib==3.9.2`
- `tensorboard==2.20.0`
- `pyglet==1.5.23`
- `pillow==12.2.0`
- `seaborn==0.13.2`
- `plotly==5.24.1`
- `pandas==2.2.3`
- `numba==0.60.0`
- `opencv-python==4.11.0.86`

这意味着当前环境已经具备：

- 训练与评估 CLI
- TensorBoard 日志读取
- 静态论文图绘制
- OpenGL/pyglet 渲染
- 视频链路所需的大部分基础组件

### 5.3 已验证的命令入口

已验证：

- `python -m swarm_rl.train --env=quadrotor_multi --help`

可以正常起 CLI，说明训练主入口没坏。

已验证：

- `python -m swarm_rl.enjoy --env=quadrotor_multi --help`

会进入评估 CLI 参数解析，并暴露 `--save_video` 等参数。

### 5.4 当前缺失项

本次确认 `swarm-rl-obstacles` 里`没有安装`：

- `imageio`
- `moviepy`

这带来的影响：

- 论文静态指标图：`不受影响`
- TensorBoard 聚合：`不受影响`
- Sample Factory 自带视频保存：`大概率不受影响`
- 自己写脚本把 `rgb_array` 帧直接合成为 GIF：`建议补装 imageio`

### 5.5 渲染侧现实限制

已有说明和代码都提示：

- 渲染相关测试需要 `X11 display`
- `human` 模式会开窗口
- `rgb_array` 模式虽然不一定弹窗，但仍走 OpenGL / framebuffer 路径

因此如果你是在纯无头终端环境里做轨迹图和视频导出，需要提前确认：

- 是否有显示环境
- 或是否要用虚拟显示方案

这不是 ROS 依赖，而是图形渲染依赖。

## 6. 是否需要 ROS / Gazebo / AirSim / 其他平台

结论：`针对你当前要做的论文仿真复现、指标图、轨迹图、动图，不需要 ROS / Gazebo / AirSim。`

本次没有在仓库中发现以下依赖链：

- `rospy`
- `roscore`
- `gazebo`
- `airsim`
- `isaac`
- `px4`
- `ardupilot`
- `webots`

当前仓库的主链路是：

- Python 仿真环境
- Sample Factory 训练/评估
- pyglet/OpenGL 渲染
- matplotlib/TensorBoard 出图

需要区分两件事：

1. `论文物理部署部分`
   - 论文中提到 `Crazyflie 2.1`、`Vicon`
   - 这是实体部署背景，不是你当前仿真复现的必需依赖

2. `当前仓库仿真复现`
   - 不依赖 ROS
   - 不依赖 Gazebo
   - 不依赖 AirSim

## 7. 当前最核心的缺口

当前能做代码与环境层面的“能力验证”，但还不能直接复现论文最终图表，原因只有一个核心缺口：

- `没有本地训练产物`

具体说就是：

1. 没有 checkpoint
2. 没有对应实验目录里的 `cfg.json`
3. 没有 TensorBoard `tfevents`
4. 因而没有 `paper/*.py` 可消费的数据根目录

所以当前阶段的判断是：

- `代码链路具备`
- `环境大体具备`
- `训练结果数据缺失`

## 8. 推荐绘图方案

下面给出一个按风险和产出效率排序的方案。

### 方案 A：先复现最小闭环，再出论文障碍主图

这是当前最合理的主方案。

#### 第一步：补齐最小缺包

建议仅补：

- `imageio`

原因：

- 静态图脚本不需要它
- 但后续自己导出 GIF 时非常方便
- 比装整套视频工具更轻

`moviepy` 不是当前必须项。

#### 第二步：先训练论文障碍基线

建议从论文最接近的 obstacle 配置开始，优先跑出：

- `8` agents
- `20%` density
- `0.6` obstacle size
- attention
- replay

优先入口：

- `swarm_rl/runs/obstacles/quads_multi_obstacles.py`

或者直接按论文风格命令行启动一个明确命名的实验目录。

产物目标：

- `train_dir/<experiment>/`
- `cfg.json`
- `checkpoint_*` 或 `best_*`
- TensorBoard `events.out.tfevents.*`

#### 第三步：先生成论文静态指标图

在有了 `tfevents` 以后，优先跑：

- `paper/mean_std_plots_quad_obstacle.py`

这可以先把障碍主实验的 success rate / collision rate / distance to goal 曲线出出来。

#### 第四步：扩展到论文其他对比图

如果要复现论文更多图，再按脚本要求组织目录：

- `paper/mean_std_plots_quad_obstacle_ablation.py`
  - 期望子目录：
    - `1_default_posxy`
    - `2_change_obs_octomap`
    - `3_add_multi_head`
    - `4_add_replay_buffer`
- `paper/mean_std_plots_quad_obstacle_num_agents.py`
  - 期望子目录：
    - `8_2`
    - `16_2`
    - `32_2`
- `paper/mean_std_plots_quad_obstacle_compare_arch_neighbor.py`
  - 期望子目录：
    - `1`
    - `2`
    - `6`
    - `16`
    - `31`
- `paper/mean_std_plots_quad_obstacle_compare_arch_density.py`
  - 期望按 obstacle density 分目录

也就是说，论文图脚本不是万能自动识别实验语义，而是`默认你先按它期待的目录名整理好实验日志`。

#### 第五步：生成轨迹图 / 视频 / GIF

推荐顺序：

1. 先用 `swarm_rl.enjoy` 加载训练好的 experiment 做可视化评估
2. 优先输出视频
3. 再从视频或帧序列转 GIF

建议理由：

- 现有 CLI 已支持 `--save_video`
- 这是仓库当前最自然的官方评估路径
- 比自己手搓渲染循环更稳

#### 第六步：生成论文 Figure 6 风格 V-value map

当 checkpoint 可加载后：

- 打开 `--visualize_v_value`
- 结合 `V_ValueMapWrapper`
- 在障碍场景评估时捕获 `rgb_array`

这样可以得到“环境截图 + V-value 图”的拼接结果。

### 方案 B：如果暂时不训练，只做仓库能力验证

如果你暂时不想开始长时间训练，那么当前能做的“非完整复现”有：

1. 跑 `paper/attn_heatmap.py`
   - 这是静态预置数据，不依赖训练日志
2. 跑 `paper/fps_compare.py`
   - 这是静态脚本，不依赖训练日志
3. 跑 `swarm_rl.train --help` / `swarm_rl.enjoy --help`
   - 验证环境入口
4. 在有显示环境时跑渲染测试
   - 验证可视化链路

但要明确：

- 这些都`不能替代`论文主结果复现
- 它们只能证明“出图/渲染代码在，环境大体可用”

## 9. 建议执行顺序

建议按这个顺序推进：

1. 在 `swarm-rl-obstacles` 中补装 `imageio`
2. 跑一个论文障碍基线训练，确保产出 `train_dir/<experiment>/`
3. 用 `swarm_rl.enjoy` 加载该实验，确认 checkpoint 可评估
4. 用 `paper/mean_std_plots_quad_obstacle.py` 先出主障碍曲线
5. 再决定是否扩展到：
   - ablation
   - varying agents
   - varying densities
   - neighbor-count comparison
6. 最后再做：
   - 轨迹视频
   - GIF
   - V-value map

## 10. 最终结论

一句话总结：

- `当前仓库没有现成的论文障碍模型或训练日志；但代码和 conda 环境已经基本具备训练、评估、静态出图和轨迹渲染能力；不需要 ROS / Gazebo / AirSim，真正缺的是本地实验产物，以及为了更方便导出 GIF 建议补一个 imageio。`

如果只看是否“现在立刻能把论文主图全部画出来”，答案是：

- `不能`

原因不是脚本缺失，而是：

- `缺少 checkpoint 和 tfevents 数据`

如果看是否“已经具备一套清晰可执行的复现与绘图路径”，答案是：

- `可以`

路径已经明确，就是：

- `先训练 -> 再 enjoy/eval -> 再用 paper 脚本聚合 -> 最后导出轨迹视频/GIF/V-value 图`
