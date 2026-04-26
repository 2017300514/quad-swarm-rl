# 中文注释副本；原始文件：gym_art/quadrotor_multi/inertia.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

#!/usr/bin/env python

# 下面开始文档字符串说明。
"""
Computing inertias of bodies.
Coordinate frame:
x - forward; y - left; z - up
The same coord frame is used for quads
All default inertias of objects are with respect to COM
Source of inertias: https://en.wikipedia.org/wiki/List_of_moments_of_inertia
WARNIGN: The h,w,l of the BoxLink are defined DIFFERENTLY compare to Wiki
"""

# 导入当前模块依赖。
import numpy as np
import copy

# 定义函数 `rotate_I`。
def rotate_I(I, R):
    # 下面开始文档字符串说明。
    """
    Rotating inertia tensor I
    R - rotation matrix
    """
    # 返回当前函数的结果。
    return R @ I @ R.T

# 定义函数 `translate_I`。
def translate_I(I, m, xyz):
    # 下面开始文档字符串说明。
    """
    Offsetting inertia tensor I by [x,y,z].T
    relative to COM
    """
    # 同时更新 `x`, `y`, `z` 等变量。
    x,y,z = xyz[0], xyz[1], xyz[2]
    # 保存或更新 `I_new` 的值。
    I_new = np.zeros([3,3])
    # 保存或更新 `I_new[0][0]` 的值。
    I_new[0][0] = I[0][0] + m * (y**2 + z**2)
    # 保存或更新 `I_new[1][1]` 的值。
    I_new[1][1] = I[1][1] + m * (x**2 + z**2)
    # 保存或更新 `I_new[2][2]` 的值。
    I_new[2][2] = I[2][2] + m * (x**2 + y**2)
    # 保存或更新 `I_new[0][1]` 的值。
    I_new[0][1] = I_new[1][0] = I[0][1] + m * x * y
    # 保存或更新 `I_new[0][2]` 的值。
    I_new[0][2] = I_new[2][0] = I[0][1] + m * x * z
    # 保存或更新 `I_new[1][2]` 的值。
    I_new[1][2] = I_new[2][1] = I[1][2] + m * y * z
    # 返回当前函数的结果。
    return I_new

# 定义函数 `deg2rad`。
def deg2rad(deg):
    # 返回当前函数的结果。
    return deg / 180. * np.pi

# 定义类 `SphereLink`。
class SphereLink():
    # 下面开始文档字符串说明。
    """
    Box object
    """
    # 保存或更新 `type` 的值。
    type = "sphere"
    # 定义函数 `__init__`。
    def __init__(self, r, m=None, density=None, name="sphere"):
        # 下面开始文档字符串说明。
        """
        m = mass
        dx = dy = dz = diameter = 2 * r
        """
        # 保存或更新 `name` 的值。
        self.name = name
        # 保存或更新 `r` 的值。
        self.r = r
        # 根据条件决定是否进入当前分支。
        if m is None:
            # 保存或更新 `m` 的值。
            self.m = self.compute_m(density)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `m` 的值。
            self.m = m
    # 为下面的函数或方法附加装饰器行为。
    @property
    # 定义函数 `I_com`。
    def I_com(self):
        # 保存或更新 `r` 的值。
        r = self.r
        # 返回当前函数的结果。
        return np.array([
            [2/5. * self.m * r **2, 0., 0.],
            [0., 2/5. * self.m * r **2, 0.],
            [0., 0., 2/5. * self.m * r **2],
        ])

    # 定义函数 `compute_m`。
    def compute_m(self, density):
        # 返回当前函数的结果。
        return density * 4./3. * np.pi * self.r ** 3


# 定义类 `BoxLink`。
class BoxLink():
    # 下面开始文档字符串说明。
    """
    Box object
    """
    # 保存或更新 `type` 的值。
    type = "box"
    # 定义函数 `__init__`。
    def __init__(self, l, w, h, m=None, density=None, name="box"):
        # 下面开始文档字符串说明。
        """
        m = mass
        dx = length = l
        dy = width = l
        dz = height = h
        """
        # 保存或更新 `name` 的值。
        self.name = name
        # 同时更新 `l`, `w`, `h` 等变量。
        self.l, self.w, self.h = l, w, h
        # 根据条件决定是否进入当前分支。
        if m is None:
            # 保存或更新 `m` 的值。
            self.m = self.compute_m(density)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `m` 的值。
            self.m = m
    # 为下面的函数或方法附加装饰器行为。
    @property
    # 定义函数 `I_com`。
    def I_com(self):
        # 同时更新 `l`, `w`, `h` 等变量。
        l ,w, h = self.l, self.w, self.h
        # 返回当前函数的结果。
        return np.array([
            [1/12. * self.m * (h**2 + w**2), 0., 0.],
            [0., 1/12. * self.m * (l**2 +  h**2), 0.],
            [0., 0., 1/12. * self.m * (w**2 + l**2)],
        ])
    
    # 定义函数 `compute_m`。
    def compute_m(self, density):
        # 返回当前函数的结果。
        return density * self.l * self.w * self.h

# 定义类 `RodLink`。
class RodLink():
    # 下面开始文档字符串说明。
    """
    Rod == Horizontal Cylinder
    """
    # 保存或更新 `type` 的值。
    type = "rod"
    # 定义函数 `__init__`。
    def __init__(self, l, r=0.002, m=None, density=None, name="rod"):
        # 下面开始文档字符串说明。
        """
        m = mass
        dx = length
        dy = dz = diameter == height
        """
        # 保存或更新 `name` 的值。
        self.name = name
        # 保存或更新 `l` 的值。
        self.l = l
        # 保存或更新 `r` 的值。
        self.r = r
        # 根据条件决定是否进入当前分支。
        if m is None:
            # 保存或更新 `m` 的值。
            self.m = self.compute_m(density)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `m` 的值。
            self.m = m
    # 为下面的函数或方法附加装饰器行为。
    @property
    # 定义函数 `I_com`。
    def I_com(self):
        # 返回当前函数的结果。
        return np.array([
            [1/12. * self.m * self.l**2, 0., 0.],
            [0., 0., 0.],
            [0., 0., 1/12. * self.m * self.l**2],
        ])

    # 定义函数 `compute_m`。
    def compute_m(self, density):
        # 返回当前函数的结果。
        return density * np.pi * self.l * self.r ** 2

# 定义类 `CylinderLink`。
class CylinderLink():
    # 下面开始文档字符串说明。
    """
    Vertical Cylinder
    """
    # 保存或更新 `type` 的值。
    type = "cylinder"
    # 定义函数 `__init__`。
    def __init__(self, h, r, m=None, density=None, name="cylinder"):
        # 下面开始文档字符串说明。
        """
        m = mass
        dz = height = h
        dy = dx = 2*radius = 2*r = diameter
        """
        # 保存或更新 `name` 的值。
        self.name = name
        # 同时更新 `h`, `r` 等变量。
        self.h, self.r = h, r
        # 根据条件决定是否进入当前分支。
        if m is None:
            # 保存或更新 `m` 的值。
            self.m = self.compute_m(density)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `m` 的值。
            self.m = m
    
    # 为下面的函数或方法附加装饰器行为。
    @property
    # 定义函数 `I_com`。
    def I_com(self):
        # 同时更新 `h`, `r` 等变量。
        h, r = self.h, self.r
        # 返回当前函数的结果。
        return np.array([
            [1/12. * self.m * (3*r**2 + h**2), 0., 0.],
            [0., 1/12. * self.m * (3*r**2 + h**2), 0.],
            [0., 0., 0.5 * self.m * r**2],
        ])
    # 定义函数 `compute_m`。
    def compute_m(self, density):
        # 返回当前函数的结果。
        return density * np.pi * self.h * self.r ** 2

# 定义类 `LinkPose`。
class LinkPose(object):
    # 定义函数 `__init__`。
    def __init__(self, R=None, xyz=None, alpha=None):
        # 下面开始文档字符串说明。
        """
        One can provide either:
        R - rotation matrix or 
        alpha - angle of roation in a xy (horizontal) plane [degrees]
        xyz - offset
        """
        # 根据条件决定是否进入当前分支。
        if xyz is not None:
            # 保存或更新 `xyz` 的值。
            self.xyz = np.array(xyz)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `xyz` 的值。
            self.xyz = np.zeros(3)
        # 根据条件决定是否进入当前分支。
        if R is not None:
            # 保存或更新 `R` 的值。
            self.R = R
        # 当上一分支不满足时，继续判断新的条件。
        elif alpha:
            # 保存或更新 `R` 的值。
            self.R = np.array([
                [np.cos(alpha), -np.sin(alpha), 0.],
                [np.sin(alpha), np.cos(alpha), 0.],
                [0., 0., 1.]
            ])
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `R` 的值。
            self.R = np.eye(3)


# 定义类 `QuadLink`。
class QuadLink(object):
    # 下面开始文档字符串说明。
    """
    Quadrotor link set to compute inertia.
    Initial coordinate system assumes being in the middle of the central body.
    Orientation of axes: x - forward; y - left; z - up
    arm_angle == |/ , i.e. between the x axis and the axis of the arm
    Quadrotor assumes X configuration.
    """
    # 定义函数 `__init__`。
    def __init__(self, params=None, verbose=False):
        # PARAMETERS (CrazyFlie by default)
        # 保存或更新 `motors_num` 的值。
        self.motors_num = 4
        # 保存或更新 `params` 的值。
        self.params = {}
        # 保存或更新 `params[body]` 的值。
        self.params["body"] = {"l": 0.03, "w": 0.03, "h": 0.004, "m": 0.005}
        # 保存或更新 `params[payload]` 的值。
        self.params["payload"] = {"l": 0.035, "w": 0.02, "h": 0.008, "m": 0.01}
        # 保存或更新 `params[arms]` 的值。
        self.params["arms"] = {"w":0.005, "h":0.005, "m":0.001}
        # 保存或更新 `params[motors]` 的值。
        self.params["motors"] = {"h":0.02, "r":0.0035, "m":0.0015}
        # 保存或更新 `params[propellers]` 的值。
        self.params["propellers"] = {"h":0.002, "r":0.022, "m":0.00075}

        # 保存或更新 `params[arms_pos]` 的值。
        self.params["arms_pos"] = {"angle": 45., "z": 0.}

        # 保存或更新 `params[payload_pos]` 的值。
        self.params["payload_pos"] = {"xy": [0., 0.], "z_sign": 1.}
        # 保存或更新 `params[motor_pos]` 的值。
        self.params["motor_pos"] = {"xyz": [0.065/2, 0.065/2, 0.]}
        # 根据条件决定是否进入当前分支。
        if params is not None:
            # 调用 `update` 执行当前处理。
            self.params.update(params)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 调用 `print` 执行当前处理。
            print("WARN: since params is None the CrazyFlie params will be used")

        # Printing all params
        # 根据条件决定是否进入当前分支。
        if verbose:
            # 调用 `print` 执行当前处理。
            print("######################################################")
            # 调用 `print` 执行当前处理。
            print("QUAD PARAMETERS:")
            # 执行这一行逻辑。
            [print(key,":", val) for key,val in self.params.items()]
            # 调用 `print` 执行当前处理。
            print("######################################################")
        
        # Dependent parameters
        # 保存或更新 `arm_angle` 的值。
        self.arm_angle = deg2rad(self.params["arms_pos"]["angle"])
        # 根据条件决定是否进入当前分支。
        if self.arm_angle == 0.:
            # 保存或更新 `arm_angle` 的值。
            self.arm_angle = 0.01
        # 保存或更新 `motor_xyz` 的值。
        self.motor_xyz = np.array(self.params["motor_pos"]["xyz"])
        # 保存或更新 `delta_y` 的值。
        delta_y = self.motor_xyz[1] - self.params["body"]["w"] / 2.
        # 根据条件决定是否进入当前分支。
        if "l" not in self.params["arms"]:
            # 保存或更新 `arm_length` 的值。
            self.arm_length =  delta_y / np.sin(self.arm_angle)
            # 保存或更新 `params[arms][l]` 的值。
            self.params["arms"]["l"] = self.arm_length
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `arm_length` 的值。
            self.arm_length = self.params["arms"]["l"]
        # print("Arm length: ", self.arm_length, "angle: ", self.arm_angle)

        # Vectors of coordinates of the COMs of arms, s.t. their ends will be exactly at motors locations
        # 保存或更新 `arm_xyz` 的值。
        self.arm_xyz = np.array([ self.motor_xyz[0] - delta_y /(2 * np.tan(self.arm_angle)),
                                 self.motor_xyz[1] - delta_y / 2,
                                 self.params["arms_pos"]["z"] ])
        

        # X signs according to clockwise starting front-left
        # i.e. the list bodies are counting clockwise: front_right, back_right, back_left, front_left
        # See CrazyFlie doc for more details: https://wiki.bitcraze.io/projects:crazyflie2:userguide:assembly
        # 保存或更新 `x_sign` 的值。
        self.x_sign = np.array([1, -1, -1, 1])
        # 保存或更新 `y_sign` 的值。
        self.y_sign = np.array([-1, -1, 1, 1])
        # 保存或更新 `sign_mx` 的值。
        self.sign_mx = np.array([self.x_sign, self.y_sign, np.array([1., 1., 1., 1.])])
        # 保存或更新 `motors_coord` 的值。
        self.motors_coord = self.sign_mx * self.motor_xyz[:, None]
        # 保存或更新 `props_coord` 的值。
        self.props_coord = copy.deepcopy(self.motors_coord)
        # 保存或更新 `props_coord[2,:]` 的值。
        self.props_coord[2,:] = (self.props_coord[2,:] + self.params["motors"]["h"] / 2. + self.params["propellers"]["h"])
        # 保存或更新 `arm_angles` 的值。
        self.arm_angles = [
            -self.arm_angle, 
             self.arm_angle, 
            -self.arm_angle, 
             self.arm_angle]
        # 保存或更新 `arms_coord` 的值。
        self.arms_coord = self.sign_mx * self.arm_xyz[:, None]

        # First defining the bodies
        # 保存或更新 `body` 的值。
        self.body =  BoxLink(**self.params["body"], name="body") # Central body 
        # 保存或更新 `payload` 的值。
        self.payload = BoxLink(**self.params["payload"], name="payload") # Could include battery
        # 保存或更新 `arms` 的值。
        self.arms  = [BoxLink(**self.params["arms"], name="arm_%d" % i) for i in range(self.motors_num)] # Just arms
        # 保存或更新 `motors` 的值。
        self.motors =  [CylinderLink(**self.params["motors"], name="motor_%d" % i) for i in range(self.motors_num)] # The motors itself
        # 保存或更新 `props` 的值。
        self.props =  [CylinderLink(**self.params["propellers"], name="prop_%d" % i) for i in range(self.motors_num)] # Propellers
        
        # 保存或更新 `links` 的值。
        self.links = [self.body, self.payload] + self.arms + self.motors + self.props

        # print("######################################################")
        # print("Inertias:")
        # [print(link.I_com, "\n") for link in self.links]
        # print("######################################################")

        # Defining locations of all bodies
        # 保存或更新 `body_pose` 的值。
        self.body_pose = LinkPose()
        # 保存或更新 `payload_pose` 的值。
        self.payload_pose = LinkPose(xyz=list(self.params["payload_pos"]["xy"]) + [np.sign(self.params["payload_pos"]["z_sign"])*(self.body.h + self.payload.h) / 2])
        # 保存或更新 `arms_pose` 的值。
        self.arms_pose = [LinkPose(alpha=self.arm_angles[i], xyz=self.arms_coord[:, i]) 
                            for i in range(self.motors_num)]
        # 保存或更新 `motors_pos` 的值。
        self.motors_pos = [LinkPose(xyz=self.motors_coord[:, i]) 
                            for i in range(self.motors_num)]
        # 保存或更新 `props_pos` 的值。
        self.props_pos = [LinkPose(xyz=self.props_coord[:, i]) 
                    for i in range(self.motors_num)]
        
        # 保存或更新 `poses` 的值。
        self.poses = [self.body_pose, self.payload_pose] + self.arms_pose + self.motors_pos + self.props_pos

        # Recomputing the center of mass of the new system of bodies
        # 保存或更新 `masses` 的值。
        masses = [link.m for link in self.links]
        # 保存或更新 `com` 的值。
        self.com = sum([ masses[i] * pose.xyz for i, pose in enumerate(self.poses)]) / self.m

        # Recomputing corrections on posess with the respect to the new system
        # self.poses_init = ujson.loads(ujson.dumps(self.poses))
        # 保存或更新 `poses_init` 的值。
        self.poses_init = copy.deepcopy(self.poses)
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for pose in self.poses:
            # 保存或更新 `pose.xyz` 的值。
            pose.xyz -= self.com
        
        # 根据条件决定是否进入当前分支。
        if verbose:
            # 调用 `print` 执行当前处理。
            print("Initial poses: ")
            # 执行这一行逻辑。
            [print(pose.xyz) for pose in self.poses_init]
            # 调用 `print` 执行当前处理。
            print("###################################")
            # 调用 `print` 执行当前处理。
            print("Final poses: ")
            # 执行这一行逻辑。
            [print(pose.xyz) for pose in self.poses]
            # 调用 `print` 执行当前处理。
            print("###################################")

        # Computing inertias
        # 保存或更新 `links_I` 的值。
        self.links_I = []
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for link_i, link in enumerate(self.links):
            # 保存或更新 `I_rot` 的值。
            I_rot = rotate_I(I=link.I_com, R=self.poses[link_i].R)
            # 保存或更新 `I_trans` 的值。
            I_trans = translate_I(I=I_rot, m=link.m, xyz=self.poses[link_i].xyz)
            # 调用 `append` 执行当前处理。
            self.links_I.append(I_trans)
        
        # Total inertia
        # 保存或更新 `I_com` 的值。
        self.I_com = sum(self.links_I)

        # Propeller poses
        # 保存或更新 `prop_pos` 的值。
        self.prop_pos = np.array([pose.xyz for pose in self.motors_pos])
    
    # 为下面的函数或方法附加装饰器行为。
    @property
    # 定义函数 `m`。
    def m(self):
        # 返回当前函数的结果。
        return np.sum([link.m for link in self.links]) 

# 定义类 `QuadLinkSimplified`。
class QuadLinkSimplified(object):
    # 下面开始文档字符串说明。
    """
    Simplified version of a quad rotor model.
    Consists of only two rods and 4 propellers, which do not contribute to any mass
    Quadrotor link set to compute inertia.
    Initial coordinate system assumes being in the middle of the central body.
    Orientation of axes: x - forward; y - left; z - up
    arm_angle == |/ , i.e. between the x axis and the axis of the arm
    Quadrotor assumes X configuration.
    """
    # 定义函数 `__init__`。
    def __init__(self, params=None, verbose=False):
        # PARAMETERS (CrazyFlie by default)
        # 保存或更新 `motors_num` 的值。
        self.motors_num = 4
        # 保存或更新 `rods_num` 的值。
        self.rods_num = 2
        # 保存或更新 `params` 的值。
        self.params = {}

        # 保存或更新 `params[body]` 的值。
        self.params["body"] = {"l": 0.03, "w": 0.03, "h": 0.004, "m": 0.005}
        # 保存或更新 `params[payload]` 的值。
        self.params["payload"] = {"l": 0.035, "w": 0.02, "h": 0.008, "m": 0.01}
        # 保存或更新 `params[arms]` 的值。
        self.params["arms"] = {"w":0.005, "h":0.005, "m":0.001}
        # 保存或更新 `params[motors]` 的值。
        self.params["motors"] = {"h":0.02, "r":0.0035, "m":0.0015}
        # 保存或更新 `params[propellers]` 的值。
        self.params["propellers"] = {"h":0.002, "r":0.022, "m":0.00075}

        # 保存或更新 `params[arms_pos]` 的值。
        self.params["arms_pos"] = {"angle": 45., "z": 0.}

        # 保存或更新 `params[payload_pos]` 的值。
        self.params["payload_pos"] = {"xy": [0., 0.], "z_sign": 1.}
        # 保存或更新 `params[motor_pos]` 的值。
        self.params["motor_pos"] = {"xyz": [0.065/2, 0.065/2, 0.]}
        # 根据条件决定是否进入当前分支。
        if params is not None:
            # 调用 `update` 执行当前处理。
            self.params.update(params)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 调用 `print` 执行当前处理。
            print("WARN: since params is None the CrazyFlie params will be used")

        ## Simplify the model
        ## arm length here represents the diagonal motor to motor distance
        # 保存或更新 `arm_length` 的值。
        self.arm_length = np.sqrt(self.params["motor_pos"]["xyz"][0]**2 * 2) * 2  # 0.092 [m]
        # 保存或更新 `params[propellers]` 的值。
        self.params["propellers"] = {"h": 0.002, "r": self.arm_length/4, "m": 0.0}
        # 保存或更新 `motor_pos_x` 的值。
        motor_pos_x = motor_pos_y = self.arm_length * np.sqrt(2) / 4
        # 保存或更新 `params[motor_pos][xyz]` 的值。
        self.params["motor_pos"]["xyz"] = [motor_pos_x, motor_pos_y, 0.0]
        # 保存或更新 `motor_xyz` 的值。
        self.motor_xyz = np.array(self.params["motor_pos"]["xyz"])
        ## calculate the total mass
        # 根据条件决定是否进入当前分支。
        if not "mass" in self.params:
            # 保存或更新 `mass_body` 的值。
            mass_body =  BoxLink(**self.params["body"]).m
            # 保存或更新 `mass_payload` 的值。
            mass_payload = BoxLink(**self.params["payload"]).m
            # 保存或更新 `mass_arms` 的值。
            mass_arms = np.sum([BoxLink(**self.params["arms"]).m for _ in range(self.motors_num)])
            # 保存或更新 `mass_motors` 的值。
            mass_motors = np.sum([CylinderLink(**self.params["motors"]).m for _ in range(self.motors_num)])
            # 保存或更新 `mass_props` 的值。
            mass_props = np.sum([CylinderLink(**self.params["propellers"]).m for _ in range(self.motors_num)])
    
            # 保存或更新 `params[mass]` 的值。
            self.params["mass"] = mass_body + mass_payload + mass_arms + mass_motors + mass_props # 0.027 [kg]

        # 保存或更新 `params[arms]` 的值。
        self.params["arms"] = {"l": self.arm_length, "r": self.arm_length / 20, "m": self.params["mass"] / 2}

        # Printing all params
        # 根据条件决定是否进入当前分支。
        if verbose:
            # 调用 `print` 执行当前处理。
            print("######################################################")
            # 调用 `print` 执行当前处理。
            print("QUAD PARAMETERS:")
            # 执行这一行逻辑。
            [print(key,":", val) for key,val in self.params.items()]
            # 调用 `print` 执行当前处理。
            print("######################################################")

        # Dependent parameters
        # 保存或更新 `arm_angle` 的值。
        self.arm_angle = deg2rad(self.params["arms_pos"]["angle"])
        # 根据条件决定是否进入当前分支。
        if self.arm_angle == 0.:
            # 保存或更新 `arm_angle` 的值。
            self.arm_angle = 0.01
        # Vectors of coordinates of the COMs of arms, s.t. their ends will be exactly at motors locations
        # 保存或更新 `arm_xyz` 的值。
        self.arm_xyz = np.array([0, 0, self.params["arms_pos"]["z"] ])

        # X signs according to clockwise starting front-left
        # i.e. the list bodies are counting clockwise: front_right, back_right, back_left, front_left
        # See CrazyFlie doc for more details: https://wiki.bitcraze.io/projects:crazyflie2:userguide:assembly
        # 保存或更新 `x_sign` 的值。
        self.x_sign = np.array([1, -1, -1, 1])
        # 保存或更新 `y_sign` 的值。
        self.y_sign = np.array([-1, -1, 1, 1])
        # 保存或更新 `sign_mx` 的值。
        self.sign_mx = np.array([self.x_sign, self.y_sign, np.array([1., 1., 1., 1.])])

        # 保存或更新 `motors_coord` 的值。
        self.motors_coord = self.sign_mx * self.motor_xyz[:, None]
        # 保存或更新 `props_coord` 的值。
        self.props_coord = copy.deepcopy(self.motors_coord)
        # 保存或更新 `props_coord[2,:]` 的值。
        self.props_coord[2,:] = (self.props_coord[2,:] + self.params["arms"]["r"] / 2. + self.params["propellers"]["h"])
        # 保存或更新 `arm_angles` 的值。
        self.arm_angles = [
            -self.arm_angle, 
             self.arm_angle
        ]

        ## define the body, the body only consists of two perpendicular rods
        # self.arms  = [RodLink(**self.params["arms"], name="arm_%d" % i) for i in range(self.rods_num)]
        # 保存或更新 `arms` 的值。
        self.arms  = [RodLink(**self.params["arms"], name="arm_%d" % i) for i in range(self.rods_num)]
        # 保存或更新 `props` 的值。
        self.props =  [CylinderLink(**self.params["propellers"], name="prop_%d" % i) for i in range(self.motors_num)] # Propellers
        # 保存或更新 `links` 的值。
        self.links = self.arms + self.props

        # Defining locations of all bodies
        # 保存或更新 `arms_pose` 的值。
        self.arms_pose = [LinkPose(alpha=self.arm_angles[i], xyz=self.arm_xyz) 
                            for i in range(self.rods_num)]
        # 保存或更新 `motors_pos` 的值。
        self.motors_pos = [LinkPose(xyz=self.motors_coord[:, i]) 
                            for i in range(self.motors_num)]
        # 保存或更新 `props_pos` 的值。
        self.props_pos = [LinkPose(xyz=self.props_coord[:, i]) 
                    for i in range(self.motors_num)]
        
        # 保存或更新 `poses` 的值。
        self.poses = self.arms_pose + self.props_pos

        # Recomputing the center of mass of the new system of bodies
        # 保存或更新 `masses` 的值。
        masses = [link.m for link in self.links]
        # 保存或更新 `com` 的值。
        self.com = sum([ masses[i] * pose.xyz for i, pose in enumerate(self.poses)]) / self.m

        # Recomputing corrections on posess with the respect to the new system
        # self.poses_init = ujson.loads(ujson.dumps(self.poses))
        # 保存或更新 `poses_init` 的值。
        self.poses_init = copy.deepcopy(self.poses)
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for pose in self.poses:
            # 保存或更新 `pose.xyz` 的值。
            pose.xyz -= self.com
        
        # 根据条件决定是否进入当前分支。
        if verbose:
            # 调用 `print` 执行当前处理。
            print("Initial poses: ")
            # 执行这一行逻辑。
            [print(pose.xyz) for pose in self.poses_init]
            # 调用 `print` 执行当前处理。
            print("###################################")
            # 调用 `print` 执行当前处理。
            print("Final poses: ")
            # 执行这一行逻辑。
            [print(pose.xyz) for pose in self.poses]
            # 调用 `print` 执行当前处理。
            print("###################################")

        # Computing inertias
        # 保存或更新 `links_I` 的值。
        self.links_I = []
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for link_i, link in enumerate(self.links):
            # 保存或更新 `I_rot` 的值。
            I_rot = rotate_I(I=link.I_com, R=self.poses[link_i].R)
            # 保存或更新 `I_trans` 的值。
            I_trans = translate_I(I=I_rot, m=link.m, xyz=self.poses[link_i].xyz)
            # 调用 `append` 执行当前处理。
            self.links_I.append(I_trans)
        
        # Total inertia
        # 保存或更新 `I_com` 的值。
        self.I_com = sum(self.links_I)

        # Propeller poses
        # 保存或更新 `prop_pos` 的值。
        self.prop_pos = np.array([pose.xyz for pose in self.motors_pos])
    
    # 为下面的函数或方法附加装饰器行为。
    @property
    # 定义函数 `m`。
    def m(self):
        # 返回当前函数的结果。
        return self.params["mass"] 


# 根据条件决定是否进入当前分支。
if __name__ == "__main__":
    # 导入当前模块依赖。
    import time
    # 保存或更新 `start_time` 的值。
    start_time = time.time()
    # 导入当前模块依赖。
    import argparse
    import yaml
    from gym_art.quadrotor_multi.quad_models import *

    # 保存或更新 `parser` 的值。
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 调用 `add_argument` 执行当前处理。
    parser.add_argument(
        '-c',"--config",
        help="Config file to test"
    )
    # 保存或更新 `args` 的值。
    args = parser.parse_args()

    # 定义函数 `report`。
    def report(quad):
        # 调用 `print` 执行当前处理。
        print("Time:", time.time()-start_time)
        # 调用 `print` 执行当前处理。
        print("Quad inertia: \n", quad.I_com)
        # 调用 `print` 执行当前处理。
        print("Quad mass:", quad.m)
        # 调用 `print` 执行当前处理。
        print("Quad arm_xyz:", quad.arm_xyz)
        # 调用 `print` 执行当前处理。
        print("Quad COM: ", quad.com)
        # 调用 `print` 执行当前处理。
        print("Quad arm_length: ", quad.arm_length)
        # 调用 `print` 执行当前处理。
        print("Quad prop_pos: \n", quad.prop_pos, "shape:", quad.prop_pos.shape)

    ## CrazyFlie parameters
    # params = {}
    # params["body"] = {"l": 0.03, "w": 0.03, "h": 0.004, "m": 0.005}
    # params["payload"] = {"l": 0.035, "w": 0.02, "h": 0.008, "m": 0.01}
    # params["arms"] = {"l": 0.022, "w":0.005, "h":0.005, "m":0.001}
    # params["motors"] = {"h":0.02, "r":0.0035, "m":0.0015}
    # params["propellers"] = {"h":0.002, "r":0.022, "m":0.00075}
    
    # params["motor_pos"] = {"xyz": [0.065/2, 0.065/2, 0.]}
    # params["arms_pos"] = {"angle": 45., "z": 0.}
    # params["payload_pos"] = {"xy": [0., 0.], "z_sign": 1}
    # z_sing corresponds to location (+1 - on top of the body, -1 - on the bottom of the body)

    # 保存或更新 `quad_crazyflie` 的值。
    quad_crazyflie = QuadLink(params=crazyflie_params()["geom"], verbose=True)
    # 调用 `print` 执行当前处理。
    print("Crazyflie: ")
    # 调用 `report` 执行当前处理。
    report(quad_crazyflie)

    ## Aztec params
    # 保存或更新 `geom_params` 的值。
    geom_params = {}
    # 保存或更新 `geom_params[body]` 的值。
    geom_params["body"] = {"l": 0.1, "w": 0.1, "h": 0.085, "m": 0.5}
    # 保存或更新 `geom_params[payload]` 的值。
    geom_params["payload"] = {"l": 0.12, "w": 0.12, "h": 0.04, "m": 0.1}
    # 保存或更新 `geom_params[arms]` 的值。
    geom_params["arms"] = {"l": 0.1, "w":0.015, "h":0.015, "m":0.025} #0.17 total arm
    # 保存或更新 `geom_params[motors]` 的值。
    geom_params["motors"] = {"h":0.02, "r":0.025, "m":0.02}
    # 保存或更新 `geom_params[propellers]` 的值。
    geom_params["propellers"] = {"h":0.01, "r":0.1, "m":0.009}
    
    # 保存或更新 `geom_params[motor_pos]` 的值。
    geom_params["motor_pos"] = {"xyz": [0.12, 0.12, 0.]}
    # 保存或更新 `geom_params[arms_pos]` 的值。
    geom_params["arms_pos"] = {"angle": 45., "z": 0.}
    # 保存或更新 `geom_params[payload_pos]` 的值。
    geom_params["payload_pos"] = {"xy": [0., 0.], "z_sign": -1}

    # 保存或更新 `quad` 的值。
    quad = QuadLink(params=geom_params, verbose=True)
    # 调用 `print` 执行当前处理。
    print("Aztec: ")
    # 调用 `report` 执行当前处理。
    report(quad)

    ## Crazyflie with lowered inertia
    # 保存或更新 `quad` 的值。
    quad = QuadLink(params=crazyflie_lowinertia_params()["geom"], verbose=True)
    # 调用 `print` 执行当前处理。
    print("Crazyflie lowered inertia: ")
    # 调用 `print` 执行当前处理。
    print("factor: ", quad_crazyflie.I_com / quad.I_com)
    # 调用 `report` 执行当前处理。
    report(quad)


    ## Random params
    # 根据条件决定是否进入当前分支。
    if args.config is not None:
        # 保存或更新 `yaml_stream` 的值。
        yaml_stream = open(args.config, 'r')
        # 保存或更新 `params_load` 的值。
        params_load = yaml.load(yaml_stream)

        # 保存或更新 `quad_load` 的值。
        quad_load = QuadLink(params=params_load, verbose=True)
        # 调用 `print` 执行当前处理。
        print("Loaded quad: %s" % args.config)
        # 调用 `report` 执行当前处理。
        report(quad_load)

    ## Simplified quad link model
    # 保存或更新 `simplified_quad` 的值。
    simplified_quad = QuadLinkSimplified(verbose=True)
    # 调用 `report` 执行当前处理。
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
