# 中文注释副本；原始文件：swarm_rl/utils.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import datetime
import random


# 定义函数 `timeStamped`。
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S-{fname}'):
    # This creates a timestamped filename so we don't overwrite our good work
    # 返回当前函数的结果。
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


# 定义函数 `generate_seeds`。
def generate_seeds(num_seeds):
    # 返回当前函数的结果。
    return [random.randrange(0, 9999) for _ in range(num_seeds)]
