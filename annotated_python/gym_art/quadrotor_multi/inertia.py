#!/usr/bin/env python

# 中文注释副本；原始文件：gym_art/quadrotor_multi/inertia.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这个文件负责把一个四旋翼拆成若干刚体部件，再把它们的几何尺寸、质量、朝向和相对位置
# 汇总成整机总质量、中心质心和惯量张量。上游输入主要来自 `quad_models.py` 的机体参数模板
# 或随机化后的几何参数；下游消费者则是动力学模块，需要这些物理量来做姿态和角速度积分。

"""
Computing inertias of bodies.
Coordinate frame:
x - forward; y - left; z - up
The same coord frame is used for quads
All default inertias of objects are with respect to COM
Source of inertias: https://en.wikipedia.org/wiki/List_of_moments_of_inertia
WARNIGN: The h,w,l of the BoxLink are defined DIFFERENTLY compare to Wiki
"""

import copy
import numpy as np


def rotate_I(I, R):
    """
    Rotating inertia tensor I
    R - rotation matrix
    """
    # 单个部件的标准惯量通常先在自身主轴坐标系里给出；
    # 真正装配整机前，必须先按当前部件朝向把它旋到机体系里。
    return R @ I @ R.T


def translate_I(I, m, xyz):
    """
    Offsetting inertia tensor I by [x,y,z].T
    relative to COM
    """
    # 这里实现的是平行轴定理：即便某个部件自己的惯量张量已知，
    # 只要它不在整机质心上，就必须再加上由质量和偏移量带来的附加项。
    x, y, z = xyz[0], xyz[1], xyz[2]
    I_new = np.zeros([3, 3])
    I_new[0][0] = I[0][0] + m * (y ** 2 + z ** 2)
    I_new[1][1] = I[1][1] + m * (x ** 2 + z ** 2)
    I_new[2][2] = I[2][2] + m * (x ** 2 + y ** 2)
    I_new[0][1] = I_new[1][0] = I[0][1] + m * x * y
    I_new[0][2] = I_new[2][0] = I[0][1] + m * x * z
    I_new[1][2] = I_new[2][1] = I[1][2] + m * y * z
    return I_new


def deg2rad(deg):
    return deg / 180. * np.pi


class SphereLink:
    # 这些 `*Link` 类的作用都一样：描述一种基础刚体形状，并在其自身质心坐标系下给出解析惯量。
    type = "sphere"

    def __init__(self, r, m=None, density=None, name="sphere"):
        """
        m = mass
        dx = dy = dz = diameter = 2 * r
        """
        self.name = name
        self.r = r
        if m is None:
            self.m = self.compute_m(density)
        else:
            self.m = m

    @property
    def I_com(self):
        r = self.r
        return np.array([
            [2 / 5. * self.m * r ** 2, 0., 0.],
            [0., 2 / 5. * self.m * r ** 2, 0.],
            [0., 0., 2 / 5. * self.m * r ** 2],
        ])

    def compute_m(self, density):
        return density * 4. / 3. * np.pi * self.r ** 3


class BoxLink:
    type = "box"

    def __init__(self, l, w, h, m=None, density=None, name="box"):
        """
        m = mass
        dx = length = l
        dy = width = l
        dz = height = h
        """
        self.name = name
        self.l, self.w, self.h = l, w, h
        if m is None:
            self.m = self.compute_m(density)
        else:
            self.m = m

    @property
    def I_com(self):
        l, w, h = self.l, self.w, self.h
        return np.array([
            [1 / 12. * self.m * (h ** 2 + w ** 2), 0., 0.],
            [0., 1 / 12. * self.m * (l ** 2 + h ** 2), 0.],
            [0., 0., 1 / 12. * self.m * (w ** 2 + l ** 2)],
        ])

    def compute_m(self, density):
        return density * self.l * self.w * self.h


class RodLink:
    """Rod == Horizontal Cylinder"""
    type = "rod"

    def __init__(self, l, r=0.002, m=None, density=None, name="rod"):
        """
        m = mass
        dx = length
        dy = dz = diameter == height
        """
        self.name = name
        self.l = l
        self.r = r
        if m is None:
            self.m = self.compute_m(density)
        else:
            self.m = m

    @property
    def I_com(self):
        return np.array([
            [1 / 12. * self.m * self.l ** 2, 0., 0.],
            [0., 0., 0.],
            [0., 0., 1 / 12. * self.m * self.l ** 2],
        ])

    def compute_m(self, density):
        return density * np.pi * self.l * self.r ** 2


class CylinderLink:
    """Vertical Cylinder"""
    type = "cylinder"

    def __init__(self, h, r, m=None, density=None, name="cylinder"):
        """
        m = mass
        dz = height = h
        dy = dx = 2*radius = 2*r = diameter
        """
        self.name = name
        self.h, self.r = h, r
        if m is None:
            self.m = self.compute_m(density)
        else:
            self.m = m

    @property
    def I_com(self):
        h, r = self.h, self.r
        return np.array([
            [1 / 12. * self.m * (3 * r ** 2 + h ** 2), 0., 0.],
            [0., 1 / 12. * self.m * (3 * r ** 2 + h ** 2), 0.],
            [0., 0., 0.5 * self.m * r ** 2],
        ])

    def compute_m(self, density):
        return density * np.pi * self.h * self.r ** 2


class LinkPose(object):
    # `LinkPose` 给每个部件补上姿态和相对位移，后面的装配阶段会把形状惯量和 pose 一起使用。
    def __init__(self, R=None, xyz=None, alpha=None):
        """
        One can provide either:
        R - rotation matrix or
        alpha - angle of roation in a xy (horizontal) plane [degrees]
        xyz - offset
        """
        if xyz is not None:
            self.xyz = np.array(xyz)
        else:
            self.xyz = np.zeros(3)
        if R is not None:
            self.R = R
        elif alpha:
            self.R = np.array([
                [np.cos(alpha), -np.sin(alpha), 0.],
                [np.sin(alpha), np.cos(alpha), 0.],
                [0., 0., 1.]
            ])
        else:
            self.R = np.eye(3)


class QuadLink(object):
    """
    Quadrotor link set to compute inertia.
    Initial coordinate system assumes being in the middle of the central body.
    Orientation of axes: x - forward; y - left; z - up
    arm_angle == |/ , i.e. between the x axis and the axis of the arm
    Quadrotor assumes X configuration.
    """

    # `QuadLink` 是完整装配器：它把机身、载荷、四根臂、电机和桨叶先实例化成单独部件，
    # 再统一重算整机质心和总惯量，因此是机体参数模板真正落到动力学物理量的关键一层。
    def __init__(self, params=None, verbose=False):
        self.motors_num = 4
        self.params = {}
        self.params["body"] = {"l": 0.03, "w": 0.03, "h": 0.004, "m": 0.005}
        self.params["payload"] = {"l": 0.035, "w": 0.02, "h": 0.008, "m": 0.01}
        self.params["arms"] = {"w": 0.005, "h": 0.005, "m": 0.001}
        self.params["motors"] = {"h": 0.02, "r": 0.0035, "m": 0.0015}
        self.params["propellers"] = {"h": 0.002, "r": 0.022, "m": 0.00075}

        self.params["arms_pos"] = {"angle": 45., "z": 0.}
        self.params["payload_pos"] = {"xy": [0., 0.], "z_sign": 1.}
        self.params["motor_pos"] = {"xyz": [0.065 / 2, 0.065 / 2, 0.]}

        if params is not None:
            self.params.update(params)
        else:
            print("WARN: since params is None the CrazyFlie params will be used")

        if verbose:
            print("######################################################")
            print("QUAD PARAMETERS:")
            [print(key, ":", val) for key, val in self.params.items()]
            print("######################################################")

        self.arm_angle = deg2rad(self.params["arms_pos"]["angle"])
        if self.arm_angle == 0.:
            self.arm_angle = 0.01
        self.motor_xyz = np.array(self.params["motor_pos"]["xyz"])
        delta_y = self.motor_xyz[1] - self.params["body"]["w"] / 2.
        if "l" not in self.params["arms"]:
            # 如果臂长未显式给出，就根据机身宽度、马达位置和 arm angle 反推出来，
            # 保证几何模板和马达布局最终是一致的。
            self.arm_length = delta_y / np.sin(self.arm_angle)
            self.params["arms"]["l"] = self.arm_length
        else:
            self.arm_length = self.params["arms"]["l"]

        # 臂本身的质心并不在电机位置，而是在靠近机身的一段；
        # 这里按几何关系把每根臂的 COM 放到一个能让臂端恰好对齐电机的位置上。
        self.arm_xyz = np.array([
            self.motor_xyz[0] - delta_y / (2 * np.tan(self.arm_angle)),
            self.motor_xyz[1] - delta_y / 2,
            self.params["arms_pos"]["z"]
        ])

        # 符号矩阵把“一个象限里的基准坐标”复制成四个电机/桨/机臂的位置。
        self.x_sign = np.array([1, -1, -1, 1])
        self.y_sign = np.array([-1, -1, 1, 1])
        self.sign_mx = np.array([self.x_sign, self.y_sign, np.array([1., 1., 1., 1.])])
        self.motors_coord = self.sign_mx * self.motor_xyz[:, None]
        self.props_coord = copy.deepcopy(self.motors_coord)
        self.props_coord[2, :] = self.props_coord[2, :] + self.params["motors"]["h"] / 2. + self.params["propellers"]["h"]
        self.arm_angles = [-self.arm_angle, self.arm_angle, -self.arm_angle, self.arm_angle]
        self.arms_coord = self.sign_mx * self.arm_xyz[:, None]

        self.body = BoxLink(**self.params["body"], name="body")
        self.payload = BoxLink(**self.params["payload"], name="payload")
        self.arms = [BoxLink(**self.params["arms"], name="arm_%d" % i) for i in range(self.motors_num)]
        self.motors = [CylinderLink(**self.params["motors"], name="motor_%d" % i) for i in range(self.motors_num)]
        self.props = [CylinderLink(**self.params["propellers"], name="prop_%d" % i) for i in range(self.motors_num)]
        self.links = [self.body, self.payload] + self.arms + self.motors + self.props

        self.body_pose = LinkPose()
        self.payload_pose = LinkPose(
            xyz=list(self.params["payload_pos"]["xy"]) +
            [np.sign(self.params["payload_pos"]["z_sign"]) * (self.body.h + self.payload.h) / 2]
        )
        self.arms_pose = [LinkPose(alpha=self.arm_angles[i], xyz=self.arms_coord[:, i]) for i in range(self.motors_num)]
        self.motors_pos = [LinkPose(xyz=self.motors_coord[:, i]) for i in range(self.motors_num)]
        self.props_pos = [LinkPose(xyz=self.props_coord[:, i]) for i in range(self.motors_num)]
        self.poses = [self.body_pose, self.payload_pose] + self.arms_pose + self.motors_pos + self.props_pos

        # 先按“各部件质量加权的几何中心”求整机真正的 COM，再把所有 pose 改写成以该 COM 为原点。
        masses = [link.m for link in self.links]
        self.com = sum([masses[i] * pose.xyz for i, pose in enumerate(self.poses)]) / self.m

        self.poses_init = copy.deepcopy(self.poses)
        for pose in self.poses:
            pose.xyz -= self.com

        if verbose:
            print("Initial poses: ")
            [print(pose.xyz) for pose in self.poses_init]
            print("###################################")
            print("Final poses: ")
            [print(pose.xyz) for pose in self.poses]
            print("###################################")

        # 最终惯量的装配顺序是：部件自身惯量 -> 按姿态旋转 -> 按 COM 偏移做平行轴修正 -> 求和。
        self.links_I = []
        for link_i, link in enumerate(self.links):
            I_rot = rotate_I(I=link.I_com, R=self.poses[link_i].R)
            I_trans = translate_I(I=I_rot, m=link.m, xyz=self.poses[link_i].xyz)
            self.links_I.append(I_trans)

        self.I_com = sum(self.links_I)

        # 这里保留各个电机位置，供电机/桨相关下游模块继续使用。
        self.prop_pos = np.array([pose.xyz for pose in self.motors_pos])

    @property
    def m(self):
        return np.sum([link.m for link in self.links])


class QuadLinkSimplified(object):
    """
    Simplified version of a quad rotor model.
    Consists of only two rods and 4 propellers, which do not contribute to any mass
    Quadrotor link set to compute inertia.
    Initial coordinate system assumes being in the middle of the central body.
    Orientation of axes: x - forward; y - left; z - up
    arm_angle == |/ , i.e. between the x axis and the axis of the arm
    Quadrotor assumes X configuration.
    """

    # 这是更轻的近似模型：把复杂机身折成两根正交杆，再用无质量桨盘保留外形尺度。
    # 它适合需要近似惯量但不想精细建模每个部件的场景。
    def __init__(self, params=None, verbose=False):
        self.motors_num = 4
        self.rods_num = 2
        self.params = {}

        self.params["body"] = {"l": 0.03, "w": 0.03, "h": 0.004, "m": 0.005}
        self.params["payload"] = {"l": 0.035, "w": 0.02, "h": 0.008, "m": 0.01}
        self.params["arms"] = {"w": 0.005, "h": 0.005, "m": 0.001}
        self.params["motors"] = {"h": 0.02, "r": 0.0035, "m": 0.0015}
        self.params["propellers"] = {"h": 0.002, "r": 0.022, "m": 0.00075}

        self.params["arms_pos"] = {"angle": 45., "z": 0.}
        self.params["payload_pos"] = {"xy": [0., 0.], "z_sign": 1.}
        self.params["motor_pos"] = {"xyz": [0.065 / 2, 0.065 / 2, 0.]}
        if params is not None:
            self.params.update(params)
        else:
            print("WARN: since params is None the CrazyFlie params will be used")

        # 这里直接从对角电机距离反推出一根“等效长杆”，再把总质量折到两根杆上。
        self.arm_length = np.sqrt(self.params["motor_pos"]["xyz"][0] ** 2 * 2) * 2
        self.params["propellers"] = {"h": 0.002, "r": self.arm_length / 4, "m": 0.0}
        motor_pos_x = motor_pos_y = self.arm_length * np.sqrt(2) / 4
        self.params["motor_pos"]["xyz"] = [motor_pos_x, motor_pos_y, 0.0]
        self.motor_xyz = np.array(self.params["motor_pos"]["xyz"])

        if "mass" not in self.params:
            mass_body = BoxLink(**self.params["body"]).m
            mass_payload = BoxLink(**self.params["payload"]).m
            mass_arms = np.sum([BoxLink(**self.params["arms"]).m for _ in range(self.motors_num)])
            mass_motors = np.sum([CylinderLink(**self.params["motors"]).m for _ in range(self.motors_num)])
            mass_props = np.sum([CylinderLink(**self.params["propellers"]).m for _ in range(self.motors_num)])
            self.params["mass"] = mass_body + mass_payload + mass_arms + mass_motors + mass_props

        self.params["arms"] = {"l": self.arm_length, "r": self.arm_length / 20, "m": self.params["mass"] / 2}

        if verbose:
            print("######################################################")
            print("QUAD PARAMETERS:")
            [print(key, ":", val) for key, val in self.params.items()]
            print("######################################################")

        self.arm_angle = deg2rad(self.params["arms_pos"]["angle"])
        if self.arm_angle == 0.:
            self.arm_angle = 0.01
        self.arm_xyz = np.array([0, 0, self.params["arms_pos"]["z"]])

        self.x_sign = np.array([1, -1, -1, 1])
        self.y_sign = np.array([-1, -1, 1, 1])
        self.sign_mx = np.array([self.x_sign, self.y_sign, np.array([1., 1., 1., 1.])])

        self.motors_coord = self.sign_mx * self.motor_xyz[:, None]
        self.props_coord = copy.deepcopy(self.motors_coord)
        self.props_coord[2, :] = self.props_coord[2, :] + self.params["arms"]["r"] / 2. + self.params["propellers"]["h"]
        self.arm_angles = [-self.arm_angle, self.arm_angle]

        self.arms = [RodLink(**self.params["arms"], name="arm_%d" % i) for i in range(self.rods_num)]
        self.props = [CylinderLink(**self.params["propellers"], name="prop_%d" % i) for i in range(self.motors_num)]
        self.links = self.arms + self.props

        self.arms_pose = [LinkPose(alpha=self.arm_angles[i], xyz=self.arm_xyz) for i in range(self.rods_num)]
        self.motors_pos = [LinkPose(xyz=self.motors_coord[:, i]) for i in range(self.motors_num)]
        self.props_pos = [LinkPose(xyz=self.props_coord[:, i]) for i in range(self.motors_num)]
        self.poses = self.arms_pose + self.props_pos

        masses = [link.m for link in self.links]
        self.com = sum([masses[i] * pose.xyz for i, pose in enumerate(self.poses)]) / self.m

        self.poses_init = copy.deepcopy(self.poses)
        for pose in self.poses:
            pose.xyz -= self.com

        if verbose:
            print("Initial poses: ")
            [print(pose.xyz) for pose in self.poses_init]
            print("###################################")
            print("Final poses: ")
            [print(pose.xyz) for pose in self.poses]
            print("###################################")

        self.links_I = []
        for link_i, link in enumerate(self.links):
            I_rot = rotate_I(I=link.I_com, R=self.poses[link_i].R)
            I_trans = translate_I(I=I_rot, m=link.m, xyz=self.poses[link_i].xyz)
            self.links_I.append(I_trans)

        self.I_com = sum(self.links_I)
        self.prop_pos = np.array([pose.xyz for pose in self.motors_pos])

    @property
    def m(self):
        return self.params["mass"]


if __name__ == "__main__":
    # 文件末尾这个脚本块不是训练路径的一部分，而是作者用来快速打印不同模板下质量、惯量、COM 和桨位置的自检入口。
    import time
    start_time = time.time()
    import argparse
    import yaml
    from gym_art.quadrotor_multi.quad_models import *

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-c', "--config",
        help="Config file to test"
    )
    args = parser.parse_args()

    def report(quad):
        print("Time:", time.time() - start_time)
        print("Quad inertia: \n", quad.I_com)
        print("Quad mass:", quad.m)
        print("Quad arm_xyz:", quad.arm_xyz)
        print("Quad COM: ", quad.com)
        print("Quad arm_length: ", quad.arm_length)
        print("Quad prop_pos: \n", quad.prop_pos, "shape:", quad.prop_pos.shape)

    quad_crazyflie = QuadLink(params=crazyflie_params()["geom"], verbose=True)
    print("Crazyflie: ")
    report(quad_crazyflie)

    geom_params = {}
    geom_params["body"] = {"l": 0.1, "w": 0.1, "h": 0.085, "m": 0.5}
    geom_params["payload"] = {"l": 0.12, "w": 0.12, "h": 0.04, "m": 0.1}
    geom_params["arms"] = {"l": 0.1, "w": 0.015, "h": 0.015, "m": 0.025}
    geom_params["motors"] = {"h": 0.02, "r": 0.025, "m": 0.02}
    geom_params["propellers"] = {"h": 0.01, "r": 0.1, "m": 0.009}
    geom_params["motor_pos"] = {"xyz": [0.12, 0.12, 0.]}
    geom_params["arms_pos"] = {"angle": 45., "z": 0.}
    geom_params["payload_pos"] = {"xy": [0., 0.], "z_sign": -1}

    quad = QuadLink(params=geom_params, verbose=True)
    print("Aztec: ")
    report(quad)

    quad = QuadLink(params=crazyflie_lowinertia_params()["geom"], verbose=True)
    print("Crazyflie lowered inertia: ")
    print("factor: ", quad_crazyflie.I_com / quad.I_com)
    report(quad)

    if args.config is not None:
        yaml_stream = open(args.config, 'r')
        params_load = yaml.load(yaml_stream)

        quad_load = QuadLink(params=params_load, verbose=True)
        print("Loaded quad: %s" % args.config)
        report(quad_load)

    simplified_quad = QuadLinkSimplified(verbose=True)
    report(simplified_quad)

################################################
## BUGS

#   geom:
#    body:
#      l: 0.03606089911004016
#      w: 0.0335657274378426
#      h: 0.006102479549661156
#      m: 0.0062767052677894074
#    payload:
#      l: 0.03838432440273057
#      w: 0.023816859339232426
#      h: 0.0070768000745466235
#      m: 0.011730161186989083
#    arms:
#      l: 0.03186274210023081
#      w: 0.004925117851085423
#      h: 0.003791035497312349
#      m: 0.0006186316884703438
#    motors:
#      h: 0.014114172356543832
#      r: 0.004954090517124884
#      m: 0.00019911121838799071
#    propellers:
#      h: 0.0003473493979756195
#      r: 0.018981731819806787
#      m: 0.0010610363239981538
#    motor_pos:
#      xyz: [0.0371103 0.0300385 0.       ]
#    arms_pos:
#      angle: 0.0
#      z: 0.0
#    payload_pos:
#      xy: [0. 0.]
#      z_sign: 0.9631780811491029
#  damp:
#    vel: 0.0013677304461958925
#    omega_quadratic: 0.013233401772593535
#  noise:
#    thrust_noise_ratio: 0.00970067152725083
#  motor:
#    thrust_to_weight: 2.928398587847958
#    torque_to_thrust: 0.07219217705972501

# self.dynamics.inertia
# array([7.80390772e-06,            nan,            nan])
