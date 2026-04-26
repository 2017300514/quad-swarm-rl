import argparse
import os
import pickle
import sys
from os.path import join
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from plots.plot_utils import set_matplotlib_params, ORANGE
from sample_factory.utils.utils import ensure_dir_exists
from matplotlib.ticker import FuncFormatter

# 这个脚本把障碍物实验的 TensorBoard 日志聚合成论文曲线图。
# 上游输入是多个训练 run 目录下的 `.tfevents.*` 文件；下游产物是一个四联图 PDF，
# 展示平均目标距离、飞行完成率、机间碰撞率和机体-障碍碰撞率随训练步数的变化。

set_matplotlib_params()

PAGE_WIDTH_INCHES = 8.2
FULL_PAGE_WIDTH = 1.4 * PAGE_WIDTH_INCHES
HALF_PAGE_WIDTH = FULL_PAGE_WIDTH / 2

plt.rcParams['figure.figsize'] = (FULL_PAGE_WIDTH, 2.3)

NUM_AGENTS = 8
EPISODE_DURATION = 16
TIME_METRIC_COLLISION = 60
COLLISIONS_SCALE = ((TIME_METRIC_COLLISION / EPISODE_DURATION) / NUM_AGENTS) * 2
COLLISIONS_OBST_SCALE = ((TIME_METRIC_COLLISION / EPISODE_DURATION) / NUM_AGENTS)

CRASH_GROUND_SCALE = (-1.0 / EPISODE_DURATION)

# `PLOTS` 定义每个子图从 TensorBoard 里取哪个标量、要做什么缩放、以及最终图例含义。
PLOTS = [
    dict(
        key='0_aux/avg_rewraw_pos',
        name='Avg. distance to the target',
        label='Avg. distance, meters',
        coeff=-1.0 / EPISODE_DURATION,
        logscale=True,
        clip_min=0.2,
        y_scale_formater=[0.2, 0.5, 1.0, 2.0],
    ),
    dict(
        key='0_aux/avg_rewraw_crash',
        name='Flight performance',
        label='Fraction of the episode in the air',
        coeff=CRASH_GROUND_SCALE,
        mutate=lambda y: 1 - y,
        clip_max=1.0,
    ),
    dict(
        key='0_aux/avg_num_collisions_after_settle',
        name='Avg. collisions between drones per minute',
        label='Number of collisions',
        logscale=True,
        coeff=COLLISIONS_SCALE,
        clip_min=0.1,
    ),
    dict(
        key='0_aux/avg_num_collisions_obst_quad',
        name='Avg. collisions between the obstacle & drones per minute',
        label='Number of collisions',
        logscale=True,
        coeff=COLLISIONS_OBST_SCALE,
        clip_min=0.06,
    ),
]

PLOT_STEP = int(5e6)
TOTAL_STEP = int(1e9 + 10000)


def extract(experiments):
    # 每个 experiment 目录对应一个训练 run，这里把它们的 TensorBoard 标量都装载出来。
    scalar_accumulators = [EventAccumulator(experiment_dir).Reload().scalars for experiment_dir in experiments]

    # 排除掉没有实际标量内容的路径，避免把空目录误当成一条 run。
    scalar_accumulators = [
        scalar_accumulator for scalar_accumulator in scalar_accumulators if scalar_accumulator.Keys()
    ]

    # 为了能做多 run 统计，要求所有 run 拥有完全一致的标量 key 集合。
    all_keys = [tuple(sorted(scalar_accumulator.Keys())) for scalar_accumulator in scalar_accumulators]
    assert len(set(all_keys)) == 1, (
        "All runs need to have the same scalar keys. There are mismatches in {}".format(all_keys)
    )

    keys = all_keys[0]
    all_scalar_events_per_key = [
        [scalar_accumulator.Items(key) for scalar_accumulator in scalar_accumulators] for key in keys
    ]

    # 原始 TensorBoard step 可能不完全对齐，这里先取出每条 run 的真实 step 序列。
    x_per_key = [
        [tuple(scalar_event.step for scalar_event in sorted(scalar_events)) for scalar_events in sorted(all_scalar_events)]
        for all_scalar_events in all_scalar_events_per_key
    ]

    # 然后统一映射到固定的绘图采样步长上，保证所有 run 在同一 x 轴上可比。
    plot_step = PLOT_STEP
    all_steps_per_key = [
        [tuple(int(step_id) for step_id in range(0, TOTAL_STEP, plot_step)) for _ in sorted(all_scalar_events)]
        for all_scalar_events in all_scalar_events_per_key
    ]

    for i, all_steps in enumerate(all_steps_per_key):
        assert len(set(all_steps)) == 1, (
            "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(
                keys[i], [len(steps) for steps in all_steps]
            )
        )

    steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

    values_per_key = [
        [[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events]
        for all_scalar_events in all_scalar_events_per_key
    ]

    interpolated_keys = dict()
    for plot_idx in range(len(PLOTS)):
        key_idx = keys.index(PLOTS[plot_idx]['key'])
        values = values_per_key[key_idx]
        x = steps_per_key[key_idx]
        x_steps = x_per_key[key_idx]

        interpolated_y = [[] for _ in values]

        for run_idx in range(len(values)):
            idx = 0

            # 前两个点被丢掉，避免最开头日志尚未稳定时的插值噪声影响图形。
            tmp_min_step = min(len(x_steps[run_idx]), len(values[run_idx]))
            values[run_idx] = values[run_idx][2:tmp_min_step]
            x_steps[run_idx] = x_steps[run_idx][2:tmp_min_step]

            assert len(x_steps[run_idx]) == len(values[run_idx])
            for x_idx in x:
                while idx < len(x_steps[run_idx]) - 1 and x_steps[run_idx][idx] < x_idx:
                    idx += 1

                # 这里用简单近邻/均值方式把原始日志粗插值到统一步长，便于后续算均值和标准差带。
                if x_idx == 0:
                    interpolated_value = values[run_idx][idx]
                elif idx < len(values[run_idx]) - 1:
                    interpolated_value = (values[run_idx][idx] + values[run_idx][idx + 1]) / 2
                else:
                    interpolated_value = values[run_idx][idx]

                interpolated_y[run_idx].append(interpolated_value)
            assert len(interpolated_y[run_idx]) == len(x)

        interpolated_keys[PLOTS[plot_idx]['key']] = (x, interpolated_y)

    return interpolated_keys


def aggregate(path, subpath, experiments, ax):
    # 聚合结果会缓存成 pickle，避免每次重新读取所有 TensorBoard 事件文件。
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = join(curr_dir, 'cache')
    cache_env = join(cache_dir, subpath)

    if os.path.isdir(cache_env):
        with open(join(cache_env, f'{subpath}.pickle'), 'rb') as fobj:
            interpolated_keys = pickle.load(fobj)
    else:
        cache_env = ensure_dir_exists(cache_env)
        interpolated_keys = extract(experiments=experiments)
        with open(join(cache_env, f'{subpath}.pickle'), 'wb') as fobj:
            pickle.dump(interpolated_keys, fobj)

    for i, key in enumerate(interpolated_keys.keys()):
        plot(i, interpolated_keys[key], ax[i])


def plot(index, interpolated_key, ax):
    params = PLOTS[index]

    ax.set_title(params['name'], fontsize=8)
    ax.set_xlabel('Simulation steps')

    x, y = interpolated_key
    y_np = [np.array(yi) for yi in y]
    y_np = np.stack(y_np)

    # 某些指标如碰撞率或目标距离跨数量级变化较大，适合对数轴展示。
    logscale = params.get('logscale', False)
    if logscale:
        ax.set_yscale('log', base=2)
        ax.yaxis.set_minor_locator(ticker.NullLocator())

    def scientific(x, pos):
        return '%.2f' % x

    scientific_formatter = FuncFormatter(scientific)
    ax.yaxis.set_major_formatter(scientific_formatter)

    # `coeff` 把原始 TensorBoard 标量换算成论文使用的物理量口径，例如每分钟碰撞数或平均距离。
    coeff = params.get('coeff', 1.0)
    y_np *= coeff

    mutate = params.get('mutate', None)
    if mutate:
        for i in range(y_np.shape[1]):
            y_np[:, i] = mutate(y_np[:, i])

    # 多条 run 聚合成均值曲线和标准差阴影带。
    y_mean = np.mean(y_np, axis=0)
    y_std = np.std(y_np, axis=0)
    y_plus_std = y_mean + y_std
    y_minus_std = y_mean - y_std

    clip_max = params.get('clip_max', None)
    if clip_max:
        y_mean = np.minimum(y_mean, clip_max)
        y_plus_std = np.minimum(y_plus_std, clip_max)
        y_minus_std = np.minimum(y_minus_std, clip_max)

    clip_min = params.get('clip_min', None)
    if clip_min:
        y_mean = np.maximum(y_mean, clip_min)
        y_plus_std = np.maximum(y_plus_std, clip_min)
        y_minus_std = np.maximum(y_minus_std, clip_min)

    # x 轴统一格式化成 K/M/B，方便显示长时间训练进度。
    def mkfunc(x, pos):
        if x >= 1e9:
            return '%dB' % int(x * 1e-9)
        elif x >= 1e6:
            return '%dM' % int(x * 1e-6)
        elif x >= 1e3:
            return '%dK' % int(x * 1e-3)
        else:
            return '%d' % int(x)

    mkformatter = matplotlib.ticker.FuncFormatter(mkfunc)
    ax.xaxis.set_major_formatter(mkformatter)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.0)

    label = params.get('label')
    if label:
        ax.set_ylabel(label, fontsize=8)
        ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.tick_params(which='major', length=0)

    ax.grid(color='#B3B3B3', linestyle='-', linewidth=0.25, alpha=0.2)

    lw = 1.4
    ax.plot(x, y_mean, color=ORANGE, linewidth=lw, antialiased=True)
    ax.fill_between(x, y_minus_std, y_plus_std, color=ORANGE, alpha=0.25, antialiased=True, linewidth=0.0)


def hide_tick_spine(ax):
    # 这是旧版多子图排版留下的辅助函数，当前主流程里已不再实际调用。
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='main path for tensorboard files', default=os.getcwd())
    parser.add_argument(
        '--output',
        type=str,
        help='aggregation can be saves as tensorboard file (summary) or as table (csv)',
        default='csv',
    )

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        raise argparse.ArgumentTypeError('Parameter {} is not a valid path'.format(path))

    # 当前实现默认取输入路径下第一个子目录名作为 cache key。
    subpath = os.listdir(path)[0]
    all_experiment_dirs = []
    for filename in Path(args.path).rglob('*.tfevents.*'):
        experiment_dir = os.path.dirname(filename)
        all_experiment_dirs.append(experiment_dir)

    if args.output not in ['summary', 'csv']:
        raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(args.output))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax = (ax1, ax2, ax3, ax4)

    aggregate(path, subpath, all_experiment_dirs, ax=ax)

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    # 最终图输出到 `../final_plots/`，供论文排版或报告整理直接引用。
    plt.savefig(
        os.path.join(os.getcwd(), '../final_plots/quads_obstacle_electron.pdf'),
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.01,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
