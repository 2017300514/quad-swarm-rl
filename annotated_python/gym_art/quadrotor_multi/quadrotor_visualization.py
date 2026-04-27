# 中文注释副本；原始文件：gym_art/quadrotor_multi/quadrotor_visualization.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这个文件是“单架无人机 3D 可视化”的组装层：上游输入是单机动力学状态、goal 和机体几何模型，
# 下游则是 `rendering3d.py` 的 scene graph、FBO/window target，以及最终给人看或给 wrapper 拼接的 RGB 帧。

import copy
from numpy.linalg import norm

from gym_art.quadrotor_multi.quad_utils import *


class ChaseCamera(object):
    # 第三人称追踪相机。
    # 它不会直接驱动仿真，只是根据当前位置、速度和目标点生成一个平滑的 “look_at” 相机轨迹，
    # 让单机回放看起来像从机体后上方跟拍。
    def __init__(self, view_dist=4):
        self.view_dist = view_dist

    def reset(self, goal, pos, vel):
        self.goal = goal
        self.pos_smooth = pos
        self.vel_smooth = vel
        self.right_smooth, _ = normalize(cross(vel, npa(0, 0, 1)))

    def step(self, pos, vel):
        # 这里通过低通滤波压掉真实动力学轨迹里的高频抖动，避免可视化镜头跟得太硬、太晃。
        ap = 0.6
        av = 0.8
        ar = 0.9
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos
        self.vel_smooth = av * self.vel_smooth + (1 - av) * vel
        self.pos = pos

        veln, n = normalize(self.vel_smooth)
        self.opp = -veln
        up = npa(0, 0, 1)
        ideal_vel, _ = normalize(self.goal - self.pos_smooth)
        # 这里优先让相机朝“通向目标的理想方向”展开，而不是死盯当前速度方向，
        # 这样在刹车、悬停或转向时仍能看清 agent 和目标之间的关系。
        right, _ = normalize(cross(ideal_vel, up))
        self.right_smooth = ar * self.right_smooth + (1 - ar) * right

    def look_at(self):
        up = npa(0, 0, 1)
        eye = self.pos_smooth + self.view_dist * (self.opp + 0.3 * up)
        center = self.pos_smooth
        return eye, center, up


class SideCamera(object):
    # 侧视相机主要给垂直运动或高度控制实验用。
    # 相比追踪镜头，它基本固定在侧面，更容易观察上升/下降和高度误差。
    def __init__(self, view_dist):
        self.view_dist = view_dist

    def reset(self, goal, pos, vel):
        self.goal = goal
        self.pos_smooth = pos
        self.vel_smooth = vel
        self.right_smooth, _ = normalize(cross(vel, npa(0, 0, 1)))

    def step(self, pos, vel):
        ap = 0.6
        av = 0.999
        ar = 0.9
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos
        self.vel_smooth = av * self.vel_smooth + (1 - av) * vel

        veln, n = normalize(self.vel_smooth)
        up = npa(0, 0, 1)
        ideal_vel, _ = normalize(self.goal - self.pos_smooth)
        if True or np.abs(veln[2]) > 0.95 or n < 0.01 or np.dot(veln, ideal_vel) < 0.7:
            # look towards goal even though we are not heading there
            right, _ = normalize(cross(ideal_vel, up))
        else:
            right, _ = normalize(cross(veln, up))
        self.right_smooth = ar * self.right_smooth + (1 - ar) * right

    def look_at(self):
        up = npa(0, 0, 1)
        eye = self.pos_smooth + self.view_dist * np.array([0, 1, 0])
        center = self.pos_smooth
        return eye, center, up


# 把 `QuadLink` 机体描述翻译成 `rendering3d` 能消费的 scene graph。
# 上游给的是几何 link 列表和每个 link 的局部 pose；下游拿到的是一个可整体平移/旋转的 `Transform` 根节点。
def quadrotor_3dmodel(model, quad_id=0):
    import gym_art.quadrotor_multi.rendering3d as r3d

    # params["body"] = {"l": 0.03, "w": 0.03, "h": 0.004, "m": 0.005}
    # params["payload"] = {"l": 0.035, "w": 0.02, "h": 0.008, "m": 0.01}
    # params["arms"] = {"l": 0.022, "w":0.005, "h":0.005, "m":0.001}
    # params["motors"] = {"h":0.02, "r":0.0035, "m":0.0015}
    # params["propellers"] = {"h":0.002, "r":0.022, "m":0.00075}

    # params["motor_pos"] = {"xyz": [0.065/2, 0.065/2, 0.]}
    # params["arms_pos"] = {"angle": 45., "z": 0.}
    # params["payload_pos"] = {"xy": [0., 0.], "z_sign": 1}

    # 这里约定了不同 link 的默认颜色；propeller 会在下面被改成按 quad id 轮换的机体主色。
    link_colors = {
        "body": (0.67843137, 1., 0.18431373),
        "payload": (0., 0., 1.),
        "prop_0": (1, 0, 0), "prop_1": (0, 1, 0), "prop_2": (0, 1, 0), "prop_3": (1, 0, 0),
        "motor_0": (0, 0, 0), "motor_1": (0, 0, 0), "motor_2": (0, 0, 0), "motor_3": (0, 0, 0),
        "arm_0": (0, 0, 1), "arm_1": (0, 0, 1), "arm_2": (0, 0, 1), "arm_3": (0, 0, 1),
    }

    links = []
    for i, link in enumerate(model.links):
        xyz, R, color = model.poses[i].xyz, model.poses[i].R, link_colors[link.name]
        rot = np.eye(4)
        rot[:3, :3] = R
        if link.name[:4] == "prop":
            prop_r = link.r
            color = np.array(QUAD_COLOR[quad_id % len(QUAD_COLOR)])
        if link.type == "box":
            link_transf = r3d.transform_and_color(
                np.matmul(r3d.translate(xyz), rot), color,
                r3d.box(link.l, link.w, link.h))
        elif link.type == "cylinder":
            link_transf = r3d.transform_and_color(r3d.translate(xyz), color,
                                                  r3d.cylinder(link.r, link.h, 32))
        elif link.type == "rod":
            R_y = np.eye(4)
            R_y[:3, :3] = rpy2R(0, np.pi / 2, 0)
            xyz[0] = -link.l / 2
            link_transf = r3d.transform_and_color(
                np.matmul(rot, np.matmul(r3d.translate(xyz), R_y)), color,
                r3d.rod(link.r, link.l, 32))

        links.append(link_transf)

    # 机头方向箭头不参与动力学，只给人眼一个“朝前”的稳定参照。
    arrow = r3d.Color((0.2, 0.3, 0.9), r3d.arrow(0.05 * prop_r, 1.5 * prop_r, 16))
    links.append(arrow)

    return r3d.Transform(np.eye(4), links)


# 当上游没有提供详细 `QuadLink` 模型时，用一个示意性的 X 型无人机替代。
# 这让环境仍然可以渲染 rollout，只是损失精细几何细节。
def quadrotor_simple_3dmodel(diam):
    import gym_art.quadrotor_multi.rendering3d as r3d

    r = diam / 2
    prop_r = 0.3 * diam
    prop_h = prop_r / 15.0

    # "X" propeller configuration, start fwd left, go clockwise
    rr = r * np.sqrt(2) / 2
    deltas = ((rr, rr, 0), (rr, -rr, 0), (-rr, -rr, 0), (-rr, rr, 0))
    colors = ((1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0))

    def disc(translation, color):
        color = 0.5 * np.array(list(color)) + 0.2
        disc = r3d.transform_and_color(r3d.translate(translation), color,
                                       r3d.cylinder(prop_r, prop_h, 32))
        return disc

    props = [disc(d, c) for d, c in zip(deltas, colors)]

    arm_thicc = diam / 20.0
    arm_color = (0.6, 0.6, 0.6)
    arms = r3d.transform_and_color(
        np.matmul(r3d.translate((0, 0, -arm_thicc)), r3d.rotz(np.pi / 4)), arm_color,
        [r3d.box(diam / 10, diam, arm_thicc), r3d.box(diam, diam / 10, arm_thicc)])

    arrow = r3d.Color((0.2, 0.3, 0.9), r3d.arrow(0.12 * prop_r, 2.5 * prop_r, 16))

    bodies = props + [arms, arrow]
    return r3d.Transform(np.eye(4), bodies)


# `Quadrotor3DScene` 是单机可视化的总装配器。
# 它把动力学状态、goal、相机轨迹和 `rendering3d` 的 target/scene graph 串起来，既能开窗口回放，也能离屏导出 RGB 帧。
class Quadrotor3DScene:
    def __init__(self, w, h,
                 quad_arm=None, model=None, resizable=True, goal_diameter=None,
                 viewpoint='chase', obs_hw=(64, 64)):

        gym_art_module = __import__('gym_art.quadrotor_multi.rendering3d')
        self.r3d = gym_art_module.quadrotor_multi.rendering3d

        self.window_target = None
        self.window_w, self.window_h = w, h
        self.resizable = resizable
        self.viepoint = viewpoint
        self.obs_hw = copy.deepcopy(obs_hw)

        # 直径既决定机体示意模型大小，也决定 goal 球和相机跟拍距离。
        self.quad_arm = quad_arm
        self.model = model

        if goal_diameter:
            self.goal_forced_diameter = goal_diameter
        else:
            self.goal_forced_diameter = None
        self.update_goal_diameter()

        if self.viepoint == 'chase':
            self.chase_cam = ChaseCamera(view_dist=self.diameter * 15)
        elif self.viepoint == 'side':
            self.chase_cam = SideCamera(view_dist=self.diameter * 15)

        self.scene = None
        self.window_target = None
        self.obs_target = None
        self.video_target = None

    def update_goal_diameter(self):
        if self.quad_arm is not None:
            self.diameter = 2 * self.quad_arm
        else:
            self.diameter = 2 * np.linalg.norm(self.model.params["motor_pos"]["xyz"][:2])

        if self.goal_forced_diameter:
            self.goal_diameter = self.goal_forced_diameter
        else:
            self.goal_diameter = self.diameter

    # 只在 target 初始化或模型刷新时重建 scene graph。
    # 真正逐帧变化的位姿都通过后面的 `set_transform*` 走增量更新，而不是每帧重建 mesh。
    def _make_scene(self):
        r3d = self.r3d

        # if target is None:
        #     self.window_target = r3d.WindowTarget(self.window_w, self.window_h, resizable=self.resizable)
        #     self.obs_target = r3d.FBOTarget(self.obs_hw[0], self.obs_hw[1])
        #     self.video_target = r3d.FBOTarget(self.window_h, self.window_h)

        self.cam1p = r3d.Camera(fov=90.0)
        self.cam3p = r3d.Camera(fov=45.0)

        if self.model is not None:
            self.quad_transform = quadrotor_3dmodel(self.model)
        else:
            self.quad_transform = quadrotor_simple_3dmodel(self.diameter)
        self.have_state = False

        self.shadow_transform = r3d.transform_and_color(
            np.eye(4), (0, 0, 0, 0.4), r3d.circle(0.75 * self.diameter, 32))

        # 地板纯粹是视觉参照，帮助观察位移和阴影，不参与物理。
        floor = r3d.ProceduralTexture(r3d.random_textype(), (0.05, 0.15),
                                      r3d.rect((1000, 1000), (0, 100), (0, 100)))

        self.update_goal_diameter()
        self.chase_cam.view_dist = self.diameter * 15

        self.create_goal(goal=(0, 0, 0))

        bodies = [r3d.BackToFront([floor, self.shadow_transform]),
                  self.goal_transform, self.quad_transform] + self.goal_arrows

        world = r3d.World(bodies)
        batch = r3d.Batch()
        world.build(batch)

        self.scene = r3d.Scene(batches=[batch], bgcolor=(0, 0, 0))
        self.scene.initialize()

    # goal 被画成一个球和三根世界坐标轴箭头，方便观察“目标点在哪里”和“当前位置如何相对目标对齐”。
    def create_goal(self, goal):
        r3d = self.r3d

        ## Goal
        self.goal_transform = r3d.transform_and_color(np.eye(4),
                                                      (0.85, 0.55, 0), r3d.sphere(self.goal_diameter / 2, 18))

        goal_arr_len, goal_arr_r, goal_arr_sect = 1.5 * self.goal_diameter, 0.02 * self.goal_diameter, 10
        self.goal_arrows = []

        self.goal_arrows_rot = []
        self.goal_arrows_rot.append(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
        self.goal_arrows_rot.append(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
        self.goal_arrows_rot.append(np.eye(3))

        self.goal_arrows.append(r3d.transform_and_color(
            np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]),
            (1., 0., 0.), r3d.arrow(goal_arr_r, goal_arr_len, goal_arr_sect)))
        self.goal_arrows.append(r3d.transform_and_color(
            np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]),
            (0., 1., 0.), r3d.arrow(goal_arr_r, goal_arr_len, goal_arr_sect)))
        self.goal_arrows.append(r3d.transform_and_color(
            np.eye(4),
            (0., 0., 1.), r3d.arrow(goal_arr_r, goal_arr_len, goal_arr_sect)))

    # 逐帧只平移 goal 标记，不重建几何。
    def update_goal(self, goal):
        r3d = self.r3d

        self.goal_transform.set_transform(r3d.translate(goal[0:3]))

        self.goal_arrows[0].set_transform(r3d.trans_and_rot(goal[0:3], self.goal_arrows_rot[0]))
        self.goal_arrows[1].set_transform(r3d.trans_and_rot(goal[0:3], self.goal_arrows_rot[1]))
        self.goal_arrows[2].set_transform(r3d.trans_and_rot(goal[0:3], self.goal_arrows_rot[2]))

    # 机体拓扑变了就必须销毁旧 target 并重建 scene，因为 mesh 结构已经不是简单位姿更新能覆盖的了。
    def update_model(self, model):
        self.model = model
        if self.video_target is not None:
            self.video_target.finish()
            self.video_target = None
        if self.obs_target is not None:
            self.obs_target.finish()
            self.obs_target = None
        if self.window_target:
            self._make_scene()

    # reset 时让相机滤波器和场景中的可视化状态重新对齐到当前 episode 开头。
    def reset(self, goal, dynamics):
        self.chase_cam.reset(goal[0:3], dynamics.pos, dynamics.vel)
        self.update_state(dynamics, goal)

    # 这是单机可视化的核心桥接层：把 dynamics 的真实位姿、第一人称 look_at 和 goal 同步进 scene graph。
    def update_state(self, dynamics, goal):
        r3d = self.r3d

        if self.scene:
            self.chase_cam.step(dynamics.pos, dynamics.vel)
            self.have_state = True
            self.fpv_lookat = dynamics.look_at()

            self.update_goal(goal=goal)

            matrix = r3d.trans_and_rot(dynamics.pos, dynamics.rot)
            self.quad_transform.set_transform_nocollide(matrix)

            shadow_pos = 0 + dynamics.pos
            shadow_pos[2] = 0.001  # avoid z-fighting
            matrix = r3d.translate(shadow_pos)
            self.shadow_transform.set_transform_nocollide(matrix)

    # 第三人称渲染入口。
    # `human` 走窗口 target，`rgb_array` 走 FBO，再把 OpenGL 坐标系里的倒置图像翻回 numpy 常见朝向。
    def render_chase(self, dynamics, goal, mode="human"):
        r3d = self.r3d

        if mode == "human":
            if self.window_target is None:
                self.window_target = r3d.WindowTarget(self.window_w, self.window_h, resizable=self.resizable)
                self._make_scene()
            self.update_state(dynamics=dynamics, goal=goal)
            self.cam3p.look_at(*self.chase_cam.look_at())
            r3d.draw(self.scene, self.cam3p, self.window_target)
            return None
        elif mode == "rgb_array":
            if self.video_target is None:
                self.video_target = r3d.FBOTarget(self.window_h, self.window_h)
                self._make_scene()
            self.update_state(dynamics=dynamics, goal=goal)
            self.cam3p.look_at(*self.chase_cam.look_at())
            r3d.draw(self.scene, self.cam3p, self.video_target)
            return np.flipud(self.video_target.read())

    # 第一人称观测渲染入口。
    # 它不使用 chase camera，而是直接复用 dynamics.look_at() 生成训练时观测对应的相机位姿。
    def render_obs(self, dynamics, goal):
        r3d = self.r3d

        if self.obs_target is None:
            self.obs_target = r3d.FBOTarget(self.obs_hw[0], self.obs_hw[1])
            self._make_scene()
        self.update_state(dynamics=dynamics, goal=goal)
        self.cam1p.look_at(*self.fpv_lookat)
        r3d.draw(self.scene, self.cam1p, self.obs_target)
        return np.flipud(self.obs_target.read())
