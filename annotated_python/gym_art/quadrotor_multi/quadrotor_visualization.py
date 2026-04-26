# 中文注释副本；原始文件：gym_art/quadrotor_multi/quadrotor_visualization.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import copy
from numpy.linalg import norm

# 导入当前模块依赖。
from gym_art.quadrotor_multi.quad_utils import *


# for visualization.
# a rough attempt at a reasonable third-person camera
# that looks "over the quadrotor's shoulder" from behind
# 定义类 `ChaseCamera`。
class ChaseCamera(object):
    # 定义函数 `__init__`。
    def __init__(self, view_dist=4):
        # 保存或更新 `view_dist` 的值。
        self.view_dist = view_dist

    # 定义函数 `reset`。
    def reset(self, goal, pos, vel):
        # 保存或更新 `goal` 的值。
        self.goal = goal
        # 保存或更新 `pos_smooth` 的值。
        self.pos_smooth = pos
        # 保存或更新 `vel_smooth` 的值。
        self.vel_smooth = vel
        # 同时更新 `right_smooth`, `_` 等变量。
        self.right_smooth, _ = normalize(cross(vel, npa(0, 0, 1)))

    # 定义函数 `step`。
    def step(self, pos, vel):
        # lowpass filter
        # 保存或更新 `ap` 的值。
        ap = 0.6
        # 保存或更新 `av` 的值。
        av = 0.8
        # 保存或更新 `ar` 的值。
        ar = 0.9
        # 保存或更新 `pos_smooth` 的值。
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos
        # 保存或更新 `vel_smooth` 的值。
        self.vel_smooth = av * self.vel_smooth + (1 - av) * vel
        # 保存或更新 `pos` 的值。
        self.pos = pos

        # 同时更新 `veln`, `n` 等变量。
        veln, n = normalize(self.vel_smooth)
        # 保存或更新 `opp` 的值。
        self.opp = -veln
        # 保存或更新 `up` 的值。
        up = npa(0, 0, 1)
        # 同时更新 `ideal_vel`, `_` 等变量。
        ideal_vel, _ = normalize(self.goal - self.pos_smooth)
        # if True or np.abs(veln[2]) > 0.95 or n < 0.01 or np.dot(veln, ideal_vel) < 0.7:
        # look towards goal even though we are not heading there
        # 同时更新 `right`, `_` 等变量。
        right, _ = normalize(cross(ideal_vel, up))
        # else:
        # right, _ = normalize(cross(veln, up))
        # 保存或更新 `right_smooth` 的值。
        self.right_smooth = ar * self.right_smooth + (1 - ar) * right

    # return eye, center, up suitable for gluLookAt
    # 定义函数 `look_at`。
    def look_at(self):
        # 保存或更新 `up` 的值。
        up = npa(0, 0, 1)
        # 同时更新 `back`, `_` 等变量。
        back, _ = normalize(cross(self.right_smooth, up))
        # 同时更新 `to_eye`, `_` 等变量。
        to_eye, _ = normalize(0.9 * self.opp + 0.2 * self.right_smooth)
        # 保存或更新 `eye` 的值。
        eye = self.pos_smooth + self.view_dist * (self.opp + 0.3 * up)
        # 保存或更新 `center` 的值。
        center = self.pos_smooth
        # 返回当前函数的结果。
        return eye, center, up


# for visualization.
# In case we have vertical control only we use a side view camera
# 定义类 `SideCamera`。
class SideCamera(object):
    # 定义函数 `__init__`。
    def __init__(self, view_dist):
        # 保存或更新 `view_dist` 的值。
        self.view_dist = view_dist

    # 定义函数 `reset`。
    def reset(self, goal, pos, vel):
        # 保存或更新 `goal` 的值。
        self.goal = goal
        # 保存或更新 `pos_smooth` 的值。
        self.pos_smooth = pos
        # 保存或更新 `vel_smooth` 的值。
        self.vel_smooth = vel
        # 同时更新 `right_smooth`, `_` 等变量。
        self.right_smooth, _ = normalize(cross(vel, npa(0, 0, 1)))

    # 定义函数 `step`。
    def step(self, pos, vel):
        # lowpass filter
        # 保存或更新 `ap` 的值。
        ap = 0.6
        # 保存或更新 `av` 的值。
        av = 0.999
        # 保存或更新 `ar` 的值。
        ar = 0.9
        # 保存或更新 `pos_smooth` 的值。
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos
        # 保存或更新 `vel_smooth` 的值。
        self.vel_smooth = av * self.vel_smooth + (1 - av) * vel

        # 同时更新 `veln`, `n` 等变量。
        veln, n = normalize(self.vel_smooth)
        # 保存或更新 `up` 的值。
        up = npa(0, 0, 1)
        # 同时更新 `ideal_vel`, `_` 等变量。
        ideal_vel, _ = normalize(self.goal - self.pos_smooth)
        # 根据条件决定是否进入当前分支。
        if True or np.abs(veln[2]) > 0.95 or n < 0.01 or np.dot(veln, ideal_vel) < 0.7:
            # look towards goal even though we are not heading there
            # 同时更新 `right`, `_` 等变量。
            right, _ = normalize(cross(ideal_vel, up))
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 同时更新 `right`, `_` 等变量。
            right, _ = normalize(cross(veln, up))
        # 保存或更新 `right_smooth` 的值。
        self.right_smooth = ar * self.right_smooth + (1 - ar) * right

    # return eye, center, up suitable for gluLookAt
    # 定义函数 `look_at`。
    def look_at(self):
        # 保存或更新 `up` 的值。
        up = npa(0, 0, 1)
        # 同时更新 `back`, `_` 等变量。
        back, _ = normalize(cross(self.right_smooth, up))
        # 同时更新 `to_eye`, `_` 等变量。
        to_eye, _ = normalize(0.9 * back + 0.3 * self.right_smooth)
        # eye = self.pos_smooth + self.view_dist * (to_eye + 0.3 * up)
        # 保存或更新 `eye` 的值。
        eye = self.pos_smooth + self.view_dist * np.array([0, 1, 0])
        # 保存或更新 `center` 的值。
        center = self.pos_smooth
        # 返回当前函数的结果。
        return eye, center, up


# 定义函数 `quadrotor_3dmodel`。
def quadrotor_3dmodel(model, quad_id=0):
    # 导入当前模块依赖。
    import gym_art.quadrotor_multi.rendering3d as r3d

    # params["body"] = {"l": 0.03, "w": 0.03, "h": 0.004, "m": 0.005}
    # params["payload"] = {"l": 0.035, "w": 0.02, "h": 0.008, "m": 0.01}
    # params["arms"] = {"l": 0.022, "w":0.005, "h":0.005, "m":0.001}
    # params["motors"] = {"h":0.02, "r":0.0035, "m":0.0015}
    # params["propellers"] = {"h":0.002, "r":0.022, "m":0.00075}

    # params["motor_pos"] = {"xyz": [0.065/2, 0.065/2, 0.]}
    # params["arms_pos"] = {"angle": 45., "z": 0.}
    # params["payload_pos"] = {"xy": [0., 0.], "z_sign": 1}

    ## PROPELLERS
    # "X" propeller configuration, start fwd left, go clockwise
    # IDs: https://wiki.bitcraze.io/projects:crazyflie2:userguide:assembly
    # 保存或更新 `link_colors` 的值。
    link_colors = {
        "body": (0.67843137, 1., 0.18431373),
        "payload": (0., 0., 1.),
        "prop_0": (1, 0, 0), "prop_1": (0, 1, 0), "prop_2": (0, 1, 0), "prop_3": (1, 0, 0),
        "motor_0": (0, 0, 0), "motor_1": (0, 0, 0), "motor_2": (0, 0, 0), "motor_3": (0, 0, 0),
        "arm_0": (0, 0, 1), "arm_1": (0, 0, 1), "arm_2": (0, 0, 1), "arm_3": (0, 0, 1),
    }

    # 保存或更新 `links` 的值。
    links = []
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i, link in enumerate(model.links):
        # 同时更新 `xyz`, `R`, `color` 等变量。
        xyz, R, color = model.poses[i].xyz, model.poses[i].R, link_colors[link.name]
        # 保存或更新 `rot` 的值。
        rot = np.eye(4)
        # 保存或更新 `rot[:3, :3]` 的值。
        rot[:3, :3] = R
        # print("LINK: ", link.name, "R:", rot, end=" ")
        # 根据条件决定是否进入当前分支。
        if link.name[:4] == "prop":
            # 保存或更新 `prop_r` 的值。
            prop_r = link.r
            # 保存或更新 `color` 的值。
            color = np.array(QUAD_COLOR[quad_id % len(QUAD_COLOR)])
        # 根据条件决定是否进入当前分支。
        if link.type == "box":
            # print("Type: Box")
            # 保存或更新 `link_transf` 的值。
            link_transf = r3d.transform_and_color(
                np.matmul(r3d.translate(xyz), rot), color,
                r3d.box(link.l, link.w, link.h))
        # 当上一分支不满足时，继续判断新的条件。
        elif link.type == "cylinder":
            # print("Type: Cylinder")
            # 保存或更新 `link_transf` 的值。
            link_transf = r3d.transform_and_color(r3d.translate(xyz), color,
                                                  r3d.cylinder(link.r, link.h, 32))
        # 当上一分支不满足时，继续判断新的条件。
        elif link.type == "rod":
            # print("Type: Rod")
            # 保存或更新 `R_y` 的值。
            R_y = np.eye(4)
            # 保存或更新 `R_y[:3, :3]` 的值。
            R_y[:3, :3] = rpy2R(0, np.pi / 2, 0)
            # 保存或更新 `xyz[0]` 的值。
            xyz[0] = -link.l / 2
            # 保存或更新 `link_transf` 的值。
            link_transf = r3d.transform_and_color(
                np.matmul(rot, np.matmul(r3d.translate(xyz), R_y)), color,
                r3d.rod(link.r, link.l, 32))

        # 调用 `append` 执行当前处理。
        links.append(link_transf)

    ## ARROWS
    # 保存或更新 `arrow` 的值。
    arrow = r3d.Color((0.2, 0.3, 0.9), r3d.arrow(0.05 * prop_r, 1.5 * prop_r, 16))
    # 调用 `append` 执行当前处理。
    links.append(arrow)

    # 返回当前函数的结果。
    return r3d.Transform(np.eye(4), links)


# 定义函数 `quadrotor_simple_3dmodel`。
def quadrotor_simple_3dmodel(diam):
    # 导入当前模块依赖。
    import gym_art.quadrotor_multi.rendering3d as r3d

    # 保存或更新 `r` 的值。
    r = diam / 2
    # 保存或更新 `prop_r` 的值。
    prop_r = 0.3 * diam
    # 保存或更新 `prop_h` 的值。
    prop_h = prop_r / 15.0

    # "X" propeller configuration, start fwd left, go clockwise
    # 保存或更新 `rr` 的值。
    rr = r * np.sqrt(2) / 2
    # 保存或更新 `deltas` 的值。
    deltas = ((rr, rr, 0), (rr, -rr, 0), (-rr, -rr, 0), (-rr, rr, 0))
    # 保存或更新 `colors` 的值。
    colors = ((1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0))

    # 定义函数 `disc`。
    def disc(translation, color):
        # 保存或更新 `color` 的值。
        color = 0.5 * np.array(list(color)) + 0.2
        # 保存或更新 `disc` 的值。
        disc = r3d.transform_and_color(r3d.translate(translation), color,
                                       r3d.cylinder(prop_r, prop_h, 32))
        # 返回当前函数的结果。
        return disc

    # 保存或更新 `props` 的值。
    props = [disc(d, c) for d, c in zip(deltas, colors)]

    # 保存或更新 `arm_thicc` 的值。
    arm_thicc = diam / 20.0
    # 保存或更新 `arm_color` 的值。
    arm_color = (0.6, 0.6, 0.6)
    # 保存或更新 `arms` 的值。
    arms = r3d.transform_and_color(
        np.matmul(r3d.translate((0, 0, -arm_thicc)), r3d.rotz(np.pi / 4)), arm_color,
        [r3d.box(diam / 10, diam, arm_thicc), r3d.box(diam, diam / 10, arm_thicc)])

    # 保存或更新 `arrow` 的值。
    arrow = r3d.Color((0.2, 0.3, 0.9), r3d.arrow(0.12 * prop_r, 2.5 * prop_r, 16))

    # 保存或更新 `bodies` 的值。
    bodies = props + [arms, arrow]
    # 返回当前函数的结果。
    return r3d.Transform(np.eye(4), bodies)


# using our rendering3d.py to draw the scene in 3D.
# this class deals both with map and mapless cases.
# 定义类 `Quadrotor3DScene`。
class Quadrotor3DScene:
    # 定义函数 `__init__`。
    def __init__(self, w, h,
                 quad_arm=None, model=None, resizable=True, goal_diameter=None,
                 # 保存或更新 `viewpoint` 的值。
                 viewpoint='chase', obs_hw=(64, 64)):

        # 保存或更新 `gym_art_module` 的值。
        gym_art_module = __import__('gym_art.quadrotor_multi.rendering3d')
        # 保存或更新 `r3d` 的值。
        self.r3d = gym_art_module.quadrotor_multi.rendering3d

        # 保存或更新 `window_target` 的值。
        self.window_target = None
        # 同时更新 `window_w`, `window_h` 等变量。
        self.window_w, self.window_h = w, h
        # 保存或更新 `resizable` 的值。
        self.resizable = resizable
        # 保存或更新 `viepoint` 的值。
        self.viepoint = viewpoint
        # 保存或更新 `obs_hw` 的值。
        self.obs_hw = copy.deepcopy(obs_hw)

        # self.world_box = 40.0
        # 保存或更新 `quad_arm` 的值。
        self.quad_arm = quad_arm
        # 保存或更新 `model` 的值。
        self.model = model

        # 根据条件决定是否进入当前分支。
        if goal_diameter:
            # 保存或更新 `goal_forced_diameter` 的值。
            self.goal_forced_diameter = goal_diameter
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `goal_forced_diameter` 的值。
            self.goal_forced_diameter = None
        # 调用 `update_goal_diameter` 执行当前处理。
        self.update_goal_diameter()

        # 根据条件决定是否进入当前分支。
        if self.viepoint == 'chase':
            # 保存或更新 `chase_cam` 的值。
            self.chase_cam = ChaseCamera(view_dist=self.diameter * 15)
        # 当上一分支不满足时，继续判断新的条件。
        elif self.viepoint == 'side':
            # 保存或更新 `chase_cam` 的值。
            self.chase_cam = SideCamera(view_dist=self.diameter * 15)

        # 保存或更新 `scene` 的值。
        self.scene = None
        # 保存或更新 `window_target` 的值。
        self.window_target = None
        # 保存或更新 `obs_target` 的值。
        self.obs_target = None
        # 保存或更新 `video_target` 的值。
        self.video_target = None

    # 定义函数 `update_goal_diameter`。
    def update_goal_diameter(self):
        # 根据条件决定是否进入当前分支。
        if self.quad_arm is not None:
            # 保存或更新 `diameter` 的值。
            self.diameter = 2 * self.quad_arm
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `diameter` 的值。
            self.diameter = 2 * np.linalg.norm(self.model.params["motor_pos"]["xyz"][:2])

        # 根据条件决定是否进入当前分支。
        if self.goal_forced_diameter:
            # 保存或更新 `goal_diameter` 的值。
            self.goal_diameter = self.goal_forced_diameter
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `goal_diameter` 的值。
            self.goal_diameter = self.diameter

    # 定义函数 `_make_scene`。
    def _make_scene(self):
        # 保存或更新 `r3d` 的值。
        r3d = self.r3d

        # if target is None:
        #     self.window_target = r3d.WindowTarget(self.window_w, self.window_h, resizable=self.resizable)
        #     self.obs_target = r3d.FBOTarget(self.obs_hw[0], self.obs_hw[1])
        #     self.video_target = r3d.FBOTarget(self.window_h, self.window_h)

        # 保存或更新 `cam1p` 的值。
        self.cam1p = r3d.Camera(fov=90.0)
        # 保存或更新 `cam3p` 的值。
        self.cam3p = r3d.Camera(fov=45.0)

        # 根据条件决定是否进入当前分支。
        if self.model is not None:
            # 保存或更新 `quad_transform` 的值。
            self.quad_transform = quadrotor_3dmodel(self.model)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `quad_transform` 的值。
            self.quad_transform = quadrotor_simple_3dmodel(self.diameter)
        # 保存或更新 `have_state` 的值。
        self.have_state = False

        # 保存或更新 `shadow_transform` 的值。
        self.shadow_transform = r3d.transform_and_color(
            np.eye(4), (0, 0, 0, 0.4), r3d.circle(0.75 * self.diameter, 32))

        # TODO make floor size or walls to indicate world_box
        # 保存或更新 `floor` 的值。
        floor = r3d.ProceduralTexture(r3d.random_textype(), (0.05, 0.15),
                                      r3d.rect((1000, 1000), (0, 100), (0, 100)))

        # 调用 `update_goal_diameter` 执行当前处理。
        self.update_goal_diameter()
        # 保存或更新 `chase_cam.view_dist` 的值。
        self.chase_cam.view_dist = self.diameter * 15

        # 保存或更新 `create_goal(goal` 的值。
        self.create_goal(goal=(0, 0, 0))

        # 保存或更新 `bodies` 的值。
        bodies = [r3d.BackToFront([floor, self.shadow_transform]),
                  self.goal_transform, self.quad_transform] + self.goal_arrows

        # 保存或更新 `world` 的值。
        world = r3d.World(bodies)
        # 保存或更新 `batch` 的值。
        batch = r3d.Batch()
        # 调用 `build` 执行当前处理。
        world.build(batch)

        # 保存或更新 `scene` 的值。
        self.scene = r3d.Scene(batches=[batch], bgcolor=(0, 0, 0))
        # 调用 `initialize` 执行当前处理。
        self.scene.initialize()

    # 定义函数 `create_goal`。
    def create_goal(self, goal):
        # 保存或更新 `r3d` 的值。
        r3d = self.r3d

        ## Goal
        # 保存或更新 `goal_transform` 的值。
        self.goal_transform = r3d.transform_and_color(np.eye(4),
                                                      (0.85, 0.55, 0), r3d.sphere(self.goal_diameter / 2, 18))

        # 同时更新 `goal_arr_len`, `goal_arr_r`, `goal_arr_sect` 等变量。
        goal_arr_len, goal_arr_r, goal_arr_sect = 1.5 * self.goal_diameter, 0.02 * self.goal_diameter, 10
        # 保存或更新 `goal_arrows` 的值。
        self.goal_arrows = []

        # 保存或更新 `goal_arrows_rot` 的值。
        self.goal_arrows_rot = []
        # 调用 `append` 执行当前处理。
        self.goal_arrows_rot.append(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
        # 调用 `append` 执行当前处理。
        self.goal_arrows_rot.append(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
        # 调用 `append` 执行当前处理。
        self.goal_arrows_rot.append(np.eye(3))

        # 调用 `append` 执行当前处理。
        self.goal_arrows.append(r3d.transform_and_color(
            np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]),
            (1., 0., 0.), r3d.arrow(goal_arr_r, goal_arr_len, goal_arr_sect)))
        # 调用 `append` 执行当前处理。
        self.goal_arrows.append(r3d.transform_and_color(
            np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]),
            (0., 1., 0.), r3d.arrow(goal_arr_r, goal_arr_len, goal_arr_sect)))
        # 调用 `append` 执行当前处理。
        self.goal_arrows.append(r3d.transform_and_color(
            np.eye(4),
            (0., 0., 1.), r3d.arrow(goal_arr_r, goal_arr_len, goal_arr_sect)))

    # 定义函数 `update_goal`。
    def update_goal(self, goal):
        # 保存或更新 `r3d` 的值。
        r3d = self.r3d

        # 调用 `set_transform` 执行当前处理。
        self.goal_transform.set_transform(r3d.translate(goal[0:3]))

        # 执行这一行逻辑。
        self.goal_arrows[0].set_transform(r3d.trans_and_rot(goal[0:3], self.goal_arrows_rot[0]))
        # 执行这一行逻辑。
        self.goal_arrows[1].set_transform(r3d.trans_and_rot(goal[0:3], self.goal_arrows_rot[1]))
        # 执行这一行逻辑。
        self.goal_arrows[2].set_transform(r3d.trans_and_rot(goal[0:3], self.goal_arrows_rot[2]))

    # 定义函数 `update_model`。
    def update_model(self, model):
        # 保存或更新 `model` 的值。
        self.model = model
        # 根据条件决定是否进入当前分支。
        if self.video_target is not None:
            # 调用 `finish` 执行当前处理。
            self.video_target.finish()
            # 保存或更新 `video_target` 的值。
            self.video_target = None
        # 根据条件决定是否进入当前分支。
        if self.obs_target is not None:
            # 调用 `finish` 执行当前处理。
            self.obs_target.finish()
            # 保存或更新 `obs_target` 的值。
            self.obs_target = None
        # 根据条件决定是否进入当前分支。
        if self.window_target:
            # 调用 `_make_scene` 执行当前处理。
            self._make_scene()

    # TODO allow resampling obstacles?
    # 定义函数 `reset`。
    def reset(self, goal, dynamics):
        # 调用 `reset` 执行当前处理。
        self.chase_cam.reset(goal[0:3], dynamics.pos, dynamics.vel)
        # 调用 `update_state` 执行当前处理。
        self.update_state(dynamics, goal)

    # 定义函数 `update_state`。
    def update_state(self, dynamics, goal):
        # 保存或更新 `r3d` 的值。
        r3d = self.r3d

        # 根据条件决定是否进入当前分支。
        if self.scene:
            # 调用 `step` 执行当前处理。
            self.chase_cam.step(dynamics.pos, dynamics.vel)
            # 保存或更新 `have_state` 的值。
            self.have_state = True
            # 保存或更新 `fpv_lookat` 的值。
            self.fpv_lookat = dynamics.look_at()

            # 保存或更新 `update_goal(goal` 的值。
            self.update_goal(goal=goal)

            # 保存或更新 `matrix` 的值。
            matrix = r3d.trans_and_rot(dynamics.pos, dynamics.rot)
            # 调用 `set_transform_nocollide` 执行当前处理。
            self.quad_transform.set_transform_nocollide(matrix)

            # 保存或更新 `shadow_pos` 的值。
            shadow_pos = 0 + dynamics.pos
            # 保存或更新 `shadow_pos[2]` 的值。
            shadow_pos[2] = 0.001  # avoid z-fighting
            # 保存或更新 `matrix` 的值。
            matrix = r3d.translate(shadow_pos)
            # 调用 `set_transform_nocollide` 执行当前处理。
            self.shadow_transform.set_transform_nocollide(matrix)

    # 定义函数 `render_chase`。
    def render_chase(self, dynamics, goal, mode="human"):
        # 保存或更新 `r3d` 的值。
        r3d = self.r3d

        # 根据条件决定是否进入当前分支。
        if mode == "human":
            # 根据条件决定是否进入当前分支。
            if self.window_target is None:
                # 保存或更新 `window_target` 的值。
                self.window_target = r3d.WindowTarget(self.window_w, self.window_h, resizable=self.resizable)
                # 调用 `_make_scene` 执行当前处理。
                self._make_scene()
            # 保存或更新 `update_state(dynamics` 的值。
            self.update_state(dynamics=dynamics, goal=goal)
            # 调用 `look_at` 执行当前处理。
            self.cam3p.look_at(*self.chase_cam.look_at())
            # 调用 `draw` 执行当前处理。
            r3d.draw(self.scene, self.cam3p, self.window_target)
            # 返回当前函数的结果。
            return None
        # 当上一分支不满足时，继续判断新的条件。
        elif mode == "rgb_array":
            # 根据条件决定是否进入当前分支。
            if self.video_target is None:
                # 保存或更新 `video_target` 的值。
                self.video_target = r3d.FBOTarget(self.window_h, self.window_h)
                # 调用 `_make_scene` 执行当前处理。
                self._make_scene()
            # 保存或更新 `update_state(dynamics` 的值。
            self.update_state(dynamics=dynamics, goal=goal)
            # 调用 `look_at` 执行当前处理。
            self.cam3p.look_at(*self.chase_cam.look_at())
            # 调用 `draw` 执行当前处理。
            r3d.draw(self.scene, self.cam3p, self.video_target)
            # 返回当前函数的结果。
            return np.flipud(self.video_target.read())

    # 定义函数 `render_obs`。
    def render_obs(self, dynamics, goal):
        # 保存或更新 `r3d` 的值。
        r3d = self.r3d

        # 根据条件决定是否进入当前分支。
        if self.obs_target is None:
            # 保存或更新 `obs_target` 的值。
            self.obs_target = r3d.FBOTarget(self.obs_hw[0], self.obs_hw[1])
            # 调用 `_make_scene` 执行当前处理。
            self._make_scene()
        # 保存或更新 `update_state(dynamics` 的值。
        self.update_state(dynamics=dynamics, goal=goal)
        # 调用 `look_at` 执行当前处理。
        self.cam1p.look_at(*self.fpv_lookat)
        # 调用 `draw` 执行当前处理。
        r3d.draw(self.scene, self.cam1p, self.obs_target)
        # 返回当前函数的结果。
        return np.flipud(self.obs_target.read())
