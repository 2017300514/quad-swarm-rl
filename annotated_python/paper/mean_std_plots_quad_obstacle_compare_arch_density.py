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

from sample_factory.utils.utils import ensure_dir_exists
from matplotlib.ticker import FuncFormatter

# 这个脚本把障碍物实验按不同障碍密度分组后画成四联图。
# 它和 `mean_std_plots_quad_obstacle.py` 的主链相同，但这里横向比较的是
# 20%/40%/60%/80% 几组 obstacle density 对成功率、碰撞率和两类距离指标的影响。

PAGE_WIDTH_INCHES = 8.2
FULL_PAGE_WIDTH = 1.4 * PAGE_WIDTH_INCHES
HALF_PAGE_WIDTH = FULL_PAGE_WIDTH / 2

plt.rcParams['figure.figsize'] = (FULL_PAGE_WIDTH, 2.5)

NUM_AGENTS = 8
EPISODE_DURATION = 16
TIME_METRIC_COLLISION = 60
COLLISIONS_SCALE = ((TIME_METRIC_COLLISION / EPISODE_DURATION) / NUM_AGENTS) * 2
COLLISIONS_OBST_SCALE = ((TIME_METRIC_COLLISION / EPISODE_DURATION) / NUM_AGENTS)

CRASH_GROUND_SCALE = (-1.0 / EPISODE_DURATION)

PLOTS = [
    dict(key='metric/agent_success_rate', name='Agent success rate', label='Average rate'),
    dict(key='metric/agent_col_rate', name='Agent collision rate', label='Average rate'),
    dict(key='o_random/distance_to_goal_1s', name='Distance to goal (random)', label='Distance (m)'),
    dict(key='o_static_same_goal/distance_to_goal_1s', name='Distance to goal (same goal)', label='Distance (m)'),
]

PLOT_STEP = int(5e6)
TOTAL_STEP = int(1e9 + 10000)

COLOR = ['#4477AA', '#EE6677', '#54B345', '#CCBB44', '#AA3377']


def extract(experiments):
    # 一组 experiments 对应同一 obstacle density 下的多次训练 run。
    scalar_accumulators = [EventAccumulator(experiment_dir).Reload().scalars for experiment_dir in experiments]
    scalar_accumulators = [
        scalar_accumulator for scalar_accumulator in scalar_accumulators if scalar_accumulator.Keys()
    ]

    # 这里默认同组 run 的日志 schema 与第一条 run 对齐；源码保留了更宽松的处理方式。
    all_keys = [tuple(sorted(scalar_accumulator.Keys())) for scalar_accumulator in scalar_accumulators]
    keys = all_keys[0]
    all_scalar_events_per_key = [
        [scalar_accumulator.Items(key) for scalar_accumulator in scalar_accumulators] for key in keys
    ]

    x_per_key = [
        [tuple(scalar_event.step for scalar_event in scalar_events) for scalar_events in all_scalar_events]
        for all_scalar_events in all_scalar_events_per_key
    ]

    all_steps_per_key = [
        [tuple(int(step_id) for step_id in range(0, TOTAL_STEP, PLOT_STEP)) for _ in all_scalar_events]
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

            tmp_min_step = min(len(x_steps[run_idx]), len(values[run_idx]))
            values[run_idx] = values[run_idx][2:tmp_min_step]
            x_steps[run_idx] = x_steps[run_idx][2:tmp_min_step]

            assert len(x_steps[run_idx]) == len(values[run_idx])
            for x_idx in x:
                while idx < len(x_steps[run_idx]) - 1 and x_steps[run_idx][idx] < x_idx:
                    idx += 1

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


def aggregate(path, subpath, experiments, ax, legend_name, group_id):
    # 每个 density 目录单独缓存一次插值后的结果，避免重复预处理。
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = join(curr_dir, 'cache')
    cache_env = join(cache_dir, subpath)

    if os.path.isdir(cache_env):
        try:
            with open(join(cache_env, f'{subpath}.pickle'), 'rb') as fobj:
                interpolated_keys = pickle.load(fobj)
        except FileNotFoundError:
            interpolated_keys = extract(experiments=experiments)
    else:
        cache_env = ensure_dir_exists(cache_env)
        interpolated_keys = extract(experiments=experiments)
        with open(join(cache_env, f'{subpath}.pickle'), 'wb') as fobj:
            pickle.dump(interpolated_keys, fobj)

    for i, key in enumerate(interpolated_keys.keys()):
        plot(i, interpolated_keys[key], ax[i], True, legend_name, group_id)


def smooth(y, radius, mode='two_sided', valid_only=False):
    """
    对单条曲线做窗口平滑。

    这个函数和后面的 EMA 一起使用：先把原始日志重采样到均匀网格，再做轻量平滑，
    让不同密度组的对比图更接近论文中最终展示的曲线形态。
    """
    assert mode in ('two_sided', 'causal')
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out


def one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    """
    把不规则 TensorBoard step 序列重采样到均匀网格，并做单边 EMA。

    这里保留源码里的实现，是因为论文作图并不是直接画原始点，而是先统一时间轴，
    再把不同 run 拉到可直接叠加的平滑曲线上。
    """
    low = xolds[0] if low is None else low
    high = xolds[-1] if high is None else high

    assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
    assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
    assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))

    xolds = xolds.astype('float64')
    yolds = yolds.astype('float64')

    luoi = 0
    sum_y = 0.
    count_y = 0.
    xnews = np.linspace(low, high, n)
    decay_period = (high - low) / (n - 1) * decay_steps
    interstep_decay = np.exp(-1. / decay_steps)
    sum_ys = np.zeros_like(xnews)
    count_ys = np.zeros_like(xnews)
    for i in range(n):
        xnew = xnews[i]
        sum_y *= interstep_decay
        count_y *= interstep_decay
        while True:
            if luoi >= len(xolds):
                break
            xold = xolds[luoi]
            if xold <= xnew:
                decay = np.exp(-(xnew - xold) / decay_period)
                sum_y += decay * yolds[luoi]
                count_y += decay
                luoi += 1
            else:
                break
        sum_ys[i] = sum_y
        count_ys[i] = count_y

    ys = sum_ys / count_ys
    ys[count_ys < low_counts_threshold] = np.nan
    return xnews, ys, count_ys


def symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    # 当前版本实际上只保留单边 EMA 结果；名字延续了论文作图脚本的演化历史。
    xs, ys, count_ys = one_sided_ema(
        xolds, yolds, low, high, n, decay_steps, low_counts_threshold=0
    )
    return xs, ys, count_ys


def plot(index, interpolated_key, ax, set_xlabel, legend_name, group_id):
    params = PLOTS[index]

    ax.set_title(params['name'], fontsize=10)
    if set_xlabel:
        ax.set_xlabel('Total environment steps')

    x, y = interpolated_key
    x = np.array(x)
    y_np = [np.array(yi) for yi in y]

    # 先把每条 run 投到统一 200 点网格，再做一轮轻微平滑。
    for i, yi in enumerate(y_np):
        xnew, y_np[i], _ = symmetric_ema(x, yi, x[0], x[-1], n=200, decay_steps=1.0)

    x = xnew
    y_np = [smooth(yi, 1) for yi in y_np]
    y_np = np.stack(y_np)

    logscale = params.get('logscale', False)
    if logscale:
        ax.set_yscale('log', base=2)
        ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: '%.1f' % value))

    # 这一版图把前两个 rate 图固定到 [0, 1]，确保不同 obstacle density 的提升/退化一眼可比。
    ax.set_ylim(-0.02, 1.0 + 0.02)

    coeff = params.get('coeff', 1.0)
    y_np *= coeff

    mutate = params.get('mutate', None)
    if mutate:
        for i in range(y_np.shape[1]):
            y_np[:, i] = mutate(y_np[:, i])

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

    def mkfunc(value, pos):
        if value >= 1e9:
            return '%dB' % int(value * 1e-9)
        elif value >= 1e6:
            return '%dM' % int(value * 1e-6)
        elif value >= 1e3:
            return '%dK' % int(value * 1e-3)
        else:
            return '%d' % int(value)

    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(mkfunc))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)

    label = params.get('label')
    if label:
        ax.set_ylabel(label, fontsize=10)
        ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.tick_params(which='major', length=0)
    ax.grid(color='#B3B3B3', linestyle='-', linewidth=0.25, alpha=0.2)

    # 每个颜色对应一种 obstacle density，阴影带表达同组多次训练的方差。
    ax.plot(x, y_mean, color=COLOR[group_id], label=legend_name, linewidth=1.4, antialiased=True)
    ax.fill_between(x, y_minus_std, y_plus_std, color=COLOR[group_id], alpha=0.25)


def hide_tick_spine(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


def main():
    # 目录名默认就表示障碍密度等级，因此这里只需收集每个子目录下的事件文件。
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='main path for tensorboard files', default=os.getcwd())
    parser.add_argument(
        '--output',
        type=str,
        help='aggregation can be saved as tensorboard file (summary) or as table (csv)',
        default='csv',
    )

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        raise argparse.ArgumentTypeError('Parameter {} is not a valid path'.format(path))

    subpaths = sorted(os.listdir(path))
    legend_name = sorted(['20%', '40%', '60%', '80%'])
    all_experiment_dirs = {}
    for subpath in subpaths:
        if subpath not in all_experiment_dirs:
            all_experiment_dirs[subpath] = []

        for filename in Path(args.path + "/" + subpath).rglob('*.tfevents.*'):
            experiment_dir = os.path.dirname(filename)
            all_experiment_dirs[subpath].append(experiment_dir)

    if args.output not in ['summary', 'csv']:
        raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(args.output))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax = (ax1, ax2, ax3, ax4)

    for i in range(len(all_experiment_dirs)):
        aggregate(path, subpaths[i], all_experiment_dirs[subpaths[i]], ax, legend_name[i], i)

    handles, labels = ax[0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.07), fontsize=10)
    lgd.set_in_layout(True)

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    save_dir = os.path.join(os.getcwd(), 'final_plots')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, 'scale_obst_density.pdf'), format='pdf', bbox_inches='tight', pad_inches=0.01)

    return 0


if __name__ == '__main__':
    sys.exit(main())
