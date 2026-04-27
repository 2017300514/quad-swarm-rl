# 中文注释副本；原始文件：swarm_rl/utils.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这里放的是几段跨脚本复用的小工具：一个负责给实验产物生成带时间戳的名字，避免覆盖旧结果；
# 另一个负责批量生成随机 seed，供多 seed 训练或评估脚本快速展开实验矩阵。

import datetime
import random


def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S-{fname}'):
    # 这个函数的价值不在字符串格式化本身，而在于把“输出文件唯一化”约定集中到一个地方，
    # 这样日志、图像或导出产物默认都不会因为重名而覆盖前一次结果。
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def generate_seeds(num_seeds):
    # 这里生成的是轻量级实验 seed 列表，常见用法是外层脚本循环这些 seed 启动多次独立 run，
    # 用来估计论文图里的均值/方差，而不是只看单次训练结果。
    return [random.randrange(0, 9999) for _ in range(num_seeds)]
