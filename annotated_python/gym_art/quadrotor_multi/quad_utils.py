# 中文注释副本；原始文件：gym_art/quadrotor_multi/quad_utils.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np
import numpy.random as nr
import numba as nb
from numba import njit
from numpy.linalg import norm
from numpy import cos, sin
from scipy import spatial
from copy import deepcopy

# 保存或更新 `EPS` 的值。
EPS = 1e-5

# 保存或更新 `QUAD_COLOR` 的值。
QUAD_COLOR = (
    (1.0, 0.0, 0.0),  # red
    (1.0, 0.5, 0.0),  # orange
    (1.0, 1.0, 0.0),  # yellow
    (0.0, 1.0, 1.0),  # cyan
    (1.0, 1.0, 0.5),  # magenta
    (0.0, 0.0, 1.0),  # blue
    (0.22, 0.2, 0.47),  # purple
    (1.0, 0.0, 1.0),  # Violet
    # (0.0, 1.0, 1.0),  # cyan
    # (1.0, 0.0, 1.0),  # magenta
    # (1.0, 0.0, 1.0),  # Violet
)

# 保存或更新 `OBST_COLOR_3` 的值。
OBST_COLOR_3 = (0., 0.5, 0.)
# 保存或更新 `OBST_COLOR_4` 的值。
OBST_COLOR_4 = (0., 0.5, 0., 1.)


# 保存或更新 `QUADS_OBS_REPR` 的值。
QUADS_OBS_REPR = {
    'xyz_vxyz_R_omega': 18,
    'xyz_vxyz_R_omega_floor': 19,
    'xyz_vxyz_R_omega_wall': 24,
}

# 保存或更新 `QUADS_NEIGHBOR_OBS_TYPE` 的值。
QUADS_NEIGHBOR_OBS_TYPE = {
    'none': 0,
    'pos_vel': 6,
}

# 保存或更新 `QUADS_OBSTACLE_OBS_TYPE` 的值。
QUADS_OBSTACLE_OBS_TYPE = {
    'none': 0,
    'octomap': 9,
}


# dict pretty printing
# 定义函数 `print_dic`。
def print_dic(dic, indent=""):
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for key, item in dic.items():
        # 根据条件决定是否进入当前分支。
        if isinstance(item, dict):
            # 调用 `print` 执行当前处理。
            print(indent, key + ":")
            # 保存或更新 `print_dic(item, indent` 的值。
            print_dic(item, indent=indent + "  ")
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 调用 `print` 执行当前处理。
            print(indent, key + ":", item)


# walk dictionary
# 定义函数 `walk_dict`。
def walk_dict(node, call):
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for key, item in node.items():
        # 根据条件决定是否进入当前分支。
        if isinstance(item, dict):
            # 调用 `walk_dict` 执行当前处理。
            walk_dict(item, call)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `node[key]` 的值。
            node[key] = call(key, item)


# 定义函数 `walk_2dict`。
def walk_2dict(node1, node2, call):
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for key, item in node1.items():
        # 根据条件决定是否进入当前分支。
        if isinstance(item, dict):
            # 调用 `walk_2dict` 执行当前处理。
            walk_2dict(item, node2[key], call)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 同时更新 `node1[key]`, `node2[key]` 等变量。
            node1[key], node2[key] = call(key, item, node2[key])


# numpy's cross is really slow for some reason
# 定义函数 `cross`。
def cross(a, b):
    # 返回当前函数的结果。
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


# returns (normalized vector, original norm)
# 定义函数 `normalize`。
def normalize(x):
    # n = norm(x)
    # 保存或更新 `n` 的值。
    n = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5  # np.sqrt(np.cumsum(np.square(x)))[2]

    # 根据条件决定是否进入当前分支。
    if n < 0.00001:
        # 返回当前函数的结果。
        return x, 0
    # 返回当前函数的结果。
    return x / n, n


# 定义函数 `norm2`。
def norm2(x):
    # 返回当前函数的结果。
    return np.sum(x ** 2)


# uniformly sample from the set of all 3D rotation matrices
# 定义函数 `rand_uniform_rot3d`。
def rand_uniform_rot3d():
    # 保存或更新 `randunit` 的值。
    randunit = lambda: normalize(np.random.normal(size=(3,)))[0]
    # 保存或更新 `up` 的值。
    up = randunit()
    # 保存或更新 `fwd` 的值。
    fwd = randunit()
    # 在条件成立时持续执行下面的循环体。
    while np.dot(fwd, up) > 0.95:
        # 保存或更新 `fwd` 的值。
        fwd = randunit()
    # 同时更新 `left`, `_` 等变量。
    left, _ = normalize(cross(up, fwd))
    # import pdb; pdb.set_trace()
    # 保存或更新 `up` 的值。
    up = cross(fwd, left)
    # 保存或更新 `rot` 的值。
    rot = np.column_stack([fwd, left, up])
    # 返回当前函数的结果。
    return rot


# shorter way to construct a numpy array
# 定义函数 `npa`。
def npa(*args):
    # 返回当前函数的结果。
    return np.array(args)


# 定义函数 `clamp_norm`。
def clamp_norm(x, maxnorm):
    # n = np.linalg.norm(x)
    # n = np.sqrt(np.cumsum(np.square(x)))[2]
    # 保存或更新 `n` 的值。
    n = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5
    # 返回当前函数的结果。
    return x if n <= maxnorm else (maxnorm / n) * x


# project a vector into the x-y plane and normalize it.
# 定义函数 `to_xyhat`。
def to_xyhat(vec):
    # 保存或更新 `v` 的值。
    v = deepcopy(vec)
    # 保存或更新 `v[2]` 的值。
    v[2] = 0
    # 同时更新 `v`, `_` 等变量。
    v, _ = normalize(v)
    # 返回当前函数的结果。
    return v


# 定义函数 `log_error`。
def log_error(err_str, ):
    # 使用上下文管理器包裹后续资源操作。
    with open("/tmp/sac/errors.txt", "a") as myfile:
        # 调用 `write` 执行当前处理。
        myfile.write(err_str)
        # myfile.write('###############################################')


# 定义函数 `quat2R`。
def quat2R(qw, qx, qy, qz):
    # 保存或更新 `R` 的值。
    R = \
        # 执行这一行逻辑。
        [[1.0 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
         [2 * qx * qy + 2 * qz * qw, 1.0 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
         [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1.0 - 2 * qx ** 2 - 2 * qy ** 2]]
    # 返回当前函数的结果。
    return np.array(R)


# 保存或更新 `quat2R_numba` 的值。
quat2R_numba = njit()(quat2R)


# 定义函数 `qwxyz2R`。
def qwxyz2R(quat):
    # 返回当前函数的结果。
    return quat2R(qw=quat[0], qx=quat[1], qy=quat[2], qz=quat[3])


# 定义函数 `quatXquat`。
def quatXquat(quat, quat_theta):
    ## quat * quat_theta
    # 保存或更新 `noisy_quat` 的值。
    noisy_quat = np.zeros(4)
    # 保存或更新 `noisy_quat[0]` 的值。
    noisy_quat[0] = quat[0] * quat_theta[0] - quat[1] * quat_theta[1] - quat[2] * quat_theta[2] - quat[3] * quat_theta[
        3]
    # 保存或更新 `noisy_quat[1]` 的值。
    noisy_quat[1] = quat[0] * quat_theta[1] + quat[1] * quat_theta[0] - quat[2] * quat_theta[3] + quat[3] * quat_theta[
        2]
    # 保存或更新 `noisy_quat[2]` 的值。
    noisy_quat[2] = quat[0] * quat_theta[2] + quat[1] * quat_theta[3] + quat[2] * quat_theta[0] - quat[3] * quat_theta[
        1]
    # 保存或更新 `noisy_quat[3]` 的值。
    noisy_quat[3] = quat[0] * quat_theta[3] - quat[1] * quat_theta[2] + quat[2] * quat_theta[1] + quat[3] * quat_theta[
        0]
    # 返回当前函数的结果。
    return noisy_quat


# 保存或更新 `quatXquat_numba` 的值。
quatXquat_numba = njit()(quatXquat)


# 定义函数 `R2quat`。
def R2quat(rot):
    # print('R2quat: ', rot, type(rot))
    # 保存或更新 `R` 的值。
    R = rot.reshape([3, 3])
    # 保存或更新 `w` 的值。
    w = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    # 保存或更新 `w4` 的值。
    w4 = (4.0 * w)
    # 保存或更新 `x` 的值。
    x = (R[2, 1] - R[1, 2]) / w4
    # 保存或更新 `y` 的值。
    y = (R[0, 2] - R[2, 0]) / w4
    # 保存或更新 `z` 的值。
    z = (R[1, 0] - R[0, 1]) / w4
    # 返回当前函数的结果。
    return np.array([w, x, y, z])


# 定义函数 `rot2D`。
def rot2D(theta):
    # 保存或更新 `c` 的值。
    c = np.cos(theta)
    # 保存或更新 `s` 的值。
    s = np.sin(theta)
    # 返回当前函数的结果。
    return np.array([[c, -s], [s, c]])


# 定义函数 `rotZ`。
def rotZ(theta):
    # 保存或更新 `r` 的值。
    r = np.eye(4)
    # 保存或更新 `r[:2, :2]` 的值。
    r[:2, :2] = rot2D(theta)
    # 返回当前函数的结果。
    return r


# 定义函数 `rpy2R`。
def rpy2R(r, p, y):
    # 保存或更新 `R_x` 的值。
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(r), -np.sin(r)],
                    [0, np.sin(r), np.cos(r)]
                    ])
    # 保存或更新 `R_y` 的值。
    R_y = np.array([[np.cos(p), 0, np.sin(p)],
                    [0, 1, 0],
                    [-np.sin(p), 0, np.cos(p)]
                    ])
    # 保存或更新 `R_z` 的值。
    R_z = np.array([[np.cos(y), -np.sin(y), 0],
                    [np.sin(y), np.cos(y), 0],
                    [0, 0, 1]
                    ])

    # 保存或更新 `R` 的值。
    R = np.dot(R_z, np.dot(R_y, R_x))

    # 返回当前函数的结果。
    return R


# 定义函数 `randyaw`。
def randyaw():
    # 保存或更新 `rotz` 的值。
    rotz = np.random.uniform(-np.pi, np.pi)
    # 返回当前函数的结果。
    return rotZ(rotz)[:3, :3]


# 定义函数 `exUxe`。
def exUxe(e, U):
    # 下面开始文档字符串说明。
    """
    Cross product approximation
    exUxe = U - (U @ e) * e, where
    Args:
        e[3,1] - norm vector (assumes the same norm vector for all vectors in the batch U)
        U[3,batch_dim] - set of vectors to perform cross product on
    Returns:
        [3,batch_dim] - batch-wise cross product approximation
    """
    # 返回当前函数的结果。
    return U - (U.T @ rot_z).T * np.repeat(rot_z, U.shape[1], axis=1)


# 定义函数 `cross_vec`。
def cross_vec(v1, v2):
    # 返回当前函数的结果。
    return np.array([[0, -v1[2], v1[1]], [v1[2], 0, -v1[0]], [-v1[1], v1[0], 0]]) @ v2


# 定义函数 `cross_mx4`。
def cross_mx4(V1, V2):
    # 保存或更新 `x1` 的值。
    x1 = cross(V1[0, :], V2[0, :])
    # 保存或更新 `x2` 的值。
    x2 = cross(V1[1, :], V2[1, :])
    # 保存或更新 `x3` 的值。
    x3 = cross(V1[2, :], V2[2, :])
    # 保存或更新 `x4` 的值。
    x4 = cross(V1[3, :], V2[3, :])
    # 返回当前函数的结果。
    return np.array([x1, x2, x3, x4])


# 定义函数 `cross_vec_mx4`。
def cross_vec_mx4(V1, V2):
    # 保存或更新 `x1` 的值。
    x1 = cross(V1, V2[0, :])
    # 保存或更新 `x2` 的值。
    x2 = cross(V1, V2[1, :])
    # 保存或更新 `x3` 的值。
    x3 = cross(V1, V2[2, :])
    # 保存或更新 `x4` 的值。
    x4 = cross(V1, V2[3, :])
    # 返回当前函数的结果。
    return np.array([x1, x2, x3, x4])


# 定义函数 `dict_update_existing`。
def dict_update_existing(dic, dic_upd):
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for key in dic_upd.keys():
        # 根据条件决定是否进入当前分支。
        if isinstance(dic[key], dict):
            # 调用 `dict_update_existing` 执行当前处理。
            dict_update_existing(dic[key], dic_upd[key])
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `dic[key]` 的值。
            dic[key] = dic_upd[key]


# 定义类 `OUNoise`。
class OUNoise:
    # 下面的文档字符串用于说明当前模块或代码块。
    """Ornstein–Uhlenbeck process"""

    # 定义函数 `__init__`。
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3, use_seed=False):
        # 下面开始文档字符串说明。
        """
        @param: mu: mean of noise
        @param: theta: stabilization coeff (i.e. noise return to mean)
        @param: sigma: noise scale coeff
        @param: use_seed: set the random number generator to some specific seed for test
        """
        # 保存或更新 `action_dimension` 的值。
        self.action_dimension = action_dimension
        # 保存或更新 `mu` 的值。
        self.mu = mu
        # 保存或更新 `theta` 的值。
        self.theta = theta
        # 保存或更新 `sigma` 的值。
        self.sigma = sigma
        # 保存或更新 `state` 的值。
        self.state = np.ones(self.action_dimension) * self.mu
        # 调用 `reset` 执行当前处理。
        self.reset()
        # 根据条件决定是否进入当前分支。
        if use_seed:
            # 调用 `seed` 执行当前处理。
            nr.seed(2)

    # 定义函数 `reset`。
    def reset(self):
        # 保存或更新 `state` 的值。
        self.state = np.ones(self.action_dimension) * self.mu

    # 定义函数 `noise`。
    def noise(self):
        # 保存或更新 `x` 的值。
        x = self.state
        # 保存或更新 `dx` 的值。
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        # 保存或更新 `state` 的值。
        self.state = x + dx
        # 返回当前函数的结果。
        return self.state


# 根据条件决定是否进入当前分支。
if __name__ == "__main__":
    # Cross product test
    # 导入当前模块依赖。
    import time

    # 保存或更新 `rot_z` 的值。
    rot_z = np.array([[3], [4], [5]])
    # 保存或更新 `rot_z` 的值。
    rot_z = rot_z / np.linalg.norm(rot_z)
    # 保存或更新 `v_rotors` 的值。
    v_rotors = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6]])

    # 保存或更新 `start_time` 的值。
    start_time = time.time()
    # 保存或更新 `cr1` 的值。
    cr1 = v_rotors - (v_rotors.T @ rot_z).T * np.repeat(rot_z, 4, axis=1)
    # 调用 `print` 执行当前处理。
    print("cr1 time:", time.time() - start_time)

    # 保存或更新 `start_time` 的值。
    start_time = time.time()
    # 保存或更新 `cr2` 的值。
    cr2 = np.cross(rot_z.T, np.cross(v_rotors.T, np.repeat(rot_z, 4, axis=1).T)).T
    # 调用 `print` 执行当前处理。
    print("cr2 time:", time.time() - start_time)
    # 调用 `print` 执行当前处理。
    print("cr1 == cr2:", np.sum(cr1 - cr2) < 1e-10)
