# 中文注释副本；原始文件：gym_art/quadrotor_multi/quadrotor_multi_visualization.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这个文件是“多机 3D 可视化”的调度层：上游给它多架无人机的动力学状态、goals、碰撞和障碍物，
# 下游则是具体窗口/FBO 里的场景更新、相机切换、路径轨迹和调试用的速度/加速度箭头。

import copy
import pyglet
from gym_art.quadrotor_multi.quad_utils import *
from gym_art.quadrotor_multi.quadrotor_visualization import ChaseCamera, SideCamera, quadrotor_simple_3dmodel, \
    quadrotor_3dmodel


# `GlobalCamera` 提供一个围绕队形中心旋转的全局观察视角。
class GlobalCamera(object):
    def __init__(self, view_dist=2.0):
        self.radius = view_dist
        self.theta = np.pi / 2
        self.phi = 0.0
        self.center = np.array([0., 0., 2.])

    def reset(self, view_dist=2.0, center=np.array([0., 0., 2.])):
        self.center = center

    def step(self, center=np.array([0., 0., 2.])):
        pass

    def look_at(self):
        up = npa(0, 0, 1)
        center = self.center  # pattern center
        eye = center + self.radius * np.array(
            [np.sin(self.theta) * np.cos(self.phi), np.sin(self.theta) * np.sin(self.phi), np.cos(self.theta)])
        return eye, center, up

# 顶视镜头主要给队形和避障关系做俯视检查。
class TopDownCamera(object):
    def __init__(self, view_dist=3.0):
        self.radius = view_dist
        self.theta = np.pi / 2
        self.phi = 0.0
        self.center = np.array([0., 0., 15.])

    def reset(self, view_dist=2.0, center=np.array([0., 0., 5.])):
        self.center = np.array([0., 0., 15.])
        #self.center = center

    def step(self, center=np.array([0., 0., 2.])):
        pass

    def look_at(self):
        up = npa(0, 1, 0)
        eye = self.center  # pattern center
        center = self.center - np.array([0, 0, 2])
        center = (center/np.linalg.norm(center)) * self.radius
        return eye, center, up

# `CornerCamera` 把镜头钉在房间角落，适合固定机位回放整场实验。
class CornerCamera(object):
    def __init__(self, view_dist=4.0, room_dims=np.array([10, 10, 10]), corner_index=0):
        self.radius = view_dist
        self.theta = np.pi / 2
        self.phi = 0.0
        self.center = np.array([0., 0., 2.])
        self.corner_index = corner_index
        self.room_dims = room_dims
        if corner_index == 0:
            self.center = np.array([-self.room_dims[0] / 2, -self.room_dims[1] / 2, self.room_dims[2]])
        elif corner_index == 1:
            self.center = np.array([self.room_dims[0] / 2, -self.room_dims[1] / 2, self.room_dims[2]])
        elif corner_index == 2:
            self.center = np.array([-self.room_dims[0] / 2, self.room_dims[1] / 2, self.room_dims[2]])
        elif corner_index == 3:
            self.center = np.array([self.room_dims[0] / 2, self.room_dims[1] / 2, self.room_dims[2]])

    def reset(self, view_dist=4.0, center=None):
        if center is not None:
            self.center = center
        elif self.corner_index == 0:
            self.center = np.array([-self.room_dims[0] / 2, -self.room_dims[1] / 2, self.room_dims[2]])
        elif self.corner_index == 1:
            self.center = np.array([self.room_dims[0] / 2, -self.room_dims[1] / 2, self.room_dims[2]])
        elif self.corner_index == 2:
            self.center = np.array([-self.room_dims[0] / 2, self.room_dims[1] / 2, self.room_dims[2]])
        elif self.corner_index == 3:
            self.center = np.array([self.room_dims[0] / 2, self.room_dims[1] / 2, self.room_dims[2]])

    def step(self, center=np.array([0., 0., 2.])):
        pass

    def look_at(self):
        up = npa(0, 0, 1)
        eye = self.center  # pattern center
        center = self.center - np.array([0, 0, 2])
        center = (center/np.linalg.norm(center)) * self.radius
        return eye, center, up

# 这个镜头像“高空跟拍版 chase camera”，中心跟着某一架无人机走，但视角保持竖直俯视。
class TopDownFollowCamera(object):
    def __init__(self, view_dist=4):
        self.view_dist = view_dist

    def reset(self, goal, pos, vel):
        self.goal = goal
        self.pos_smooth = pos
        self.vel_smooth = vel
        self.right_smooth, _ = normalize(cross(vel, npa(0, 0, 1)))

    def step(self, pos, vel):
        # lowpass filter
        ap = 0.6
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos

    # return eye, center, up suitable for gluLookAt
    def look_at(self):
        up = npa(0, 1, 0)
        eye = self.pos_smooth + np.array([0, 0, 5])
        center = self.pos_smooth
        return eye, center, up


# `Quadrotor3DSceneMulti` 是多机回放的总控制器。
# 它管理所有 drone/goal/obstacle/collision/path 的 scene node，并负责键盘切视角、多窗口排布与离屏渲染。
class Quadrotor3DSceneMulti:
    first_spawn_x = 0

    def __init__(
            self, w, h,
            quad_arm=None, models=None, walls_visible=True, resizable=True, goal_diameter=None,
            viewpoint='chase', obs_hw=None, room_dims=(10, 10, 10), num_agents=8, obstacles=None,
            render_speed=1.0, formation_size=-1.0, vis_vel_arrows=True, vis_acc_arrows=True, viz_traces=100, viz_trace_nth_step=1,
            num_obstacles=0, scene_index=0
    ):
        self.pygl_window = __import__('pyglet.window', fromlist=['key'])
        self.keys = None  # keypress handler, initialized later

        if obs_hw is None:
            obs_hw = [64, 64]

        self.window_target = None
        self.window_w, self.window_h = w, h
        self.resizable = resizable
        self.viewpoint = viewpoint
        self.obs_hw = copy.deepcopy(obs_hw)
        self.walls_visible = walls_visible
        self.scene_index = scene_index

        # `diameter` 同时控制无人机示意模型、goal marker 和相机距离，是多机视觉比例尺的核心。
        self.quad_arm = quad_arm
        self.models = models
        self.room_dims = room_dims

        self.quad_transforms, self.shadow_transforms, self.goal_transforms = [], [], []

        if goal_diameter:
            self.goal_forced_diameter = goal_diameter
        else:
            self.goal_forced_diameter = None

        self.diameter = self.goal_diameter = -1
        self.update_goal_diameter()

        if self.viewpoint == 'chase':
            self.chase_cam = ChaseCamera(view_dist=self.diameter * 15)
        elif self.viewpoint == 'side':
            self.chase_cam = SideCamera(view_dist=self.diameter * 15)
        elif self.viewpoint == 'global':
            self.chase_cam = GlobalCamera(view_dist=2.5)
        elif self.viewpoint == 'topdown':
            self.chase_cam = TopDownCamera(view_dist=2.5)
        elif self.viewpoint == 'topdownfollow':
            self.chase_cam = TopDownFollowCamera(view_dist=2.5)
        elif self.viewpoint[:-1] == 'corner':
            self.chase_cam = CornerCamera(view_dist=4.0, room_dims=self.room_dims, corner_index=int(self.viewpoint[-1]))

        self.fpv_lookat = None

        self.scene = None
        self.window_target = None
        self.obs_target = None
        self.video_target = None

        self.obstacles = None
        if obstacles:
            self.obstacles = obstacles

        # 这些缓存让窗口键盘切换视角时不必向外层环境重新要状态。
        self.goals = None
        self.dynamics = None
        self.num_agents = num_agents
        self.camera_drone_index = 0

        # Aux camera moving
        standard_render_speed = 1.0
        speed_ratio = render_speed / standard_render_speed
        self.camera_rot_step_size = np.pi / 45 * speed_ratio
        self.camera_zoom_step_size = 0.1 * speed_ratio
        self.camera_mov_step_size = 0.1 * speed_ratio
        self.formation_size = formation_size
        self.vis_vel_arrows = vis_vel_arrows
        self.vis_acc_arrows = vis_acc_arrows
        self.viz_traces = 50
        self.viz_trace_nth_step = viz_trace_nth_step
        self.vector_array = [[] for _ in range(num_agents)]
        self.store_path_every_n = 1
        self.store_path_count = 0
        self.path_store = [[] for _ in range(num_agents)]

    def update_goal_diameter(self):
        if self.quad_arm is not None:
            self.diameter = self.quad_arm
        else:
            self.diameter = np.linalg.norm(self.models[0].params['motor_pos']['xyz'][:2])

        if self.goal_forced_diameter:
            self.goal_diameter = self.goal_forced_diameter
        else:
            self.goal_diameter = self.diameter

    def update_env(self, room_dims):
        self.room_dims = room_dims
        self._make_scene()

    # 构建多机场景的静态图结构。
    # 这里一次性创建所有 transform 句柄，后续 `update_state` 只改矩阵和颜色。
    def _make_scene(self):
        import gym_art.quadrotor_multi.rendering3d as r3d

        self.cam1p = r3d.Camera(fov=90.0)
        self.cam3p = r3d.Camera(fov=45.0)

        self.quad_transforms, self.shadow_transforms, self.goal_transforms, self.collision_transforms = [], [], [], []
        self.obstacle_transforms, self.vec_cyl_transforms, self.vec_cone_transforms = [], [], []
        self.path_transforms = [[] for _ in range(self.num_agents)]

        shadow_circle = r3d.circle(0.75 * self.diameter, 32)
        collision_sphere = r3d.sphere(0.75 * self.diameter, 32)

        arrow_cylinder = r3d.cylinder(0.005, 0.12, 16)
        arrow_cone = r3d.cone(0.01, 0.04, 16)
        path_sphere = r3d.sphere(0.15 * self.diameter, 16)

        for i, model in enumerate(self.models):
            if model is not None:
                quad_transform = quadrotor_3dmodel(model, quad_id=i)
            else:
                quad_transform = quadrotor_simple_3dmodel(self.diameter)
            self.quad_transforms.append(quad_transform)

            self.shadow_transforms.append(
                r3d.transform_and_color(np.eye(4), (0, 0, 0, 0.0), shadow_circle)
            )
            self.collision_transforms.append(
                r3d.transform_and_color(np.eye(4), (0, 0, 0, 0.0), collision_sphere)
            )
            if self.vis_vel_arrows:
                self.vec_cyl_transforms.append(
                    r3d.transform_and_color(np.eye(4), (1, 1, 1), arrow_cylinder)
                )
                self.vec_cone_transforms.append(
                    r3d.transform_and_color(np.eye(4), (1, 1, 1), arrow_cone)
                )
            if self.vis_acc_arrows:
                self.vec_cyl_transforms.append(
                    r3d.transform_and_color(np.eye(4), (1, 1, 1), arrow_cylinder)
                )
                self.vec_cone_transforms.append(
                    r3d.transform_and_color(np.eye(4), (1, 1, 1), arrow_cone)
                )

            if self.viz_traces:
                color = QUAD_COLOR[i % len(QUAD_COLOR)] + (1.0,)
                for j in range(self.viz_traces):
                    self.path_transforms[i].append(r3d.transform_and_color(np.eye(4), color, path_sphere))

        # 地板、墙体和障碍物都是空间参照，帮助观察队形和避障，不参与这里的状态更新逻辑。
        floor = r3d.ProceduralTexture(2, (0.85, 0.95),
                                      r3d.rect((100, 100), (0, 100), (0, 100)))
        self.update_goal_diameter()
        self.chase_cam.view_dist = self.diameter * 15

        self.create_goals()

        bodies = [r3d.BackToFront([floor, st]) for st in self.shadow_transforms]
        bodies.extend(self.goal_transforms)
        bodies.extend(self.quad_transforms)
        bodies.extend(self.vec_cyl_transforms)
        bodies.extend(self.vec_cone_transforms)
        for path in self.path_transforms:
            bodies.extend(path)
        # visualize walls of the room if True
        if self.walls_visible:
            room = r3d.ProceduralTexture(r3d.random_textype(), (0.75, 0.85), r3d.envBox(*self.room_dims))
            bodies.append(room)

        if self.obstacles:
            self.create_obstacles()
            bodies.extend(self.obstacle_transforms)

        world = r3d.World(bodies)
        batch = r3d.Batch()
        world.build(batch)
        self.scene = r3d.Scene(batches=[batch], bgcolor=(0, 0, 0))
        self.scene.initialize()

        # 碰撞球是透明层，所以单独放到最后一个 batch，避免和不透明几何的绘制顺序互相污染。
        bodies = []
        bodies.extend(self.collision_transforms)
        world = r3d.World(bodies)
        batch = r3d.Batch()
        world.build(batch)
        self.scene.batches.extend([batch])

    # 为每个 obstacle 创建一个贯穿房间高度的圆柱体，和论文里的柱状障碍布局一致。
    def create_obstacles(self):
        import gym_art.quadrotor_multi.rendering3d as r3d
        for item in self.obstacles.pos_arr:
            color = OBST_COLOR_3
            obst_height = self.room_dims[2]
            obstacle_transform = r3d.transform_and_color(np.eye(4), color, r3d.cylinder(
                radius=self.obstacles.size / 2.0, height=obst_height, sections=64))

            self.obstacle_transforms.append(obstacle_transform)

    # obstacle 拓扑默认不变，只更新位置和平移后的颜色。
    def update_obstacles(self, obstacles):
        import gym_art.quadrotor_multi.rendering3d as r3d

        if len(obstacles.pos_arr) == 1:
            return

        for i, g in enumerate(obstacles.pos_arr):
            # self.obstacle_transforms[i].set_transform(r3d.translate(g.pos))
            pos_update = [g[0], g[1], g[2] - self.room_dims[2] / 2]

            # color = QUAD_COLOR
            self.obstacle_transforms[i].set_transform_and_color(r3d.translate(pos_update), OBST_COLOR_4)

    #def create_arrows(self, envs):
    #
    #    self.

    # 每个 agent 的 goal 用一颗同色小球表示，这样颜色可以和机体一一对应。
    def create_goals(self):
        import gym_art.quadrotor_multi.rendering3d as r3d

        goal_sphere = r3d.sphere(0.1 / 2, 18)
        for i in range(len(self.models)):
            color = QUAD_COLOR[i % len(QUAD_COLOR)]
            goal_transform = r3d.transform_and_color(np.eye(4), color, goal_sphere)
            self.goal_transforms.append(goal_transform)

    # goal 每帧只做平移，不重建 marker。
    def update_goals(self, goals):
        import gym_art.quadrotor_multi.rendering3d as r3d

        for i, g in enumerate(goals):
            self.goal_transforms[i].set_transform(r3d.translate(g[0:3]))

    # 当上游替换机体模型时，scene graph 需要整体重建。
    def update_models(self, models):
        self.models = models

        if self.video_target is not None:
            self.video_target.finish()
            self.video_target = None
        if self.obs_target is not None:
            self.obs_target.finish()
            self.obs_target = None
        if self.window_target:
            self._make_scene()

    # reset 会重置轨迹缓存、相机滤波器和视角中心，让新 episode 从干净状态开始可视化。
    def reset(self, goals, dynamics, obstacles, collisions):
        self.goals = goals
        self.dynamics = dynamics
        self.vector_array = [[] for _ in range(self.num_agents)]
        self.path_store = [[] for _ in range(self.num_agents)]

        if self.viewpoint == 'global':
            goal = np.mean(goals, axis=0)
            self.chase_cam.reset(view_dist=2.5, center=goal)
        elif self.viewpoint[:-1] == 'corner' or self.viewpoint == 'topdown':
            self.chase_cam.reset()
        else:
            goal = goals[self.camera_drone_index]  # TODO: make a camera that can look at all drones
            self.chase_cam.reset(goal[0:3], dynamics[self.camera_drone_index].pos,
                                 dynamics[self.camera_drone_index].vel)

        self.update_state(dynamics, goals, obstacles, collisions)

    # 多机可视化的核心桥接层。
    # 它把每个 agent 的位姿、goal、障碍物、路径、速度/加速度和碰撞状态同步到 scene graph。
    def update_state(self, all_dynamics, goals, obstacles, collisions):
        import gym_art.quadrotor_multi.rendering3d as r3d

        if self.scene:
            if self.viewpoint == 'global' or self.viewpoint[:-1] == 'corner' or self.viewpoint == 'topdown':
                goal = np.mean(goals, axis=0)
                self.chase_cam.step(center=goal)
            else:
                self.chase_cam.step(all_dynamics[self.camera_drone_index].pos,
                                    all_dynamics[self.camera_drone_index].vel)
                self.fpv_lookat = all_dynamics[self.camera_drone_index].look_at()
            self.store_path_count += 1
            self.update_goals(goals=goals)
            if self.obstacles:
                self.update_obstacles(obstacles)

            for i, dyn in enumerate(all_dynamics):
                matrix = r3d.trans_and_rot(dyn.pos, dyn.rot)
                self.quad_transforms[i].set_transform_nocollide(matrix)

                translation = r3d.translate(dyn.pos)

                if self.viz_traces and self.store_path_count % self.viz_trace_nth_step == 0:
                    # 轨迹球不是无限追加，而是维护一个固定长度的“尾迹 ring buffer”。
                    if len(self.path_store[i]) >= self.viz_traces:
                        self.path_store[i].pop(0)

                    self.path_store[i].append(translation)
                    color_rgba = QUAD_COLOR[i % len(QUAD_COLOR)] + (1.0,)
                    path_storage_length = len(self.path_store[i])
                    for k in range(path_storage_length):
                        scale = k / path_storage_length + 0.01
                        transformation = self.path_store[i][k] @ r3d.scale(scale)
                        self.path_transforms[i][k].set_transform_and_color(transformation, color_rgba)

                if self.vis_vel_arrows:
                    if len(self.vector_array[i]) > 10:
                        self.vector_array[i].pop(0)

                    self.vector_array[i].append(dyn.vel)

                    # 箭头先做一个短窗口平均，避免瞬时速度抖动让可视化难以读。
                    avg_of_vecs = np.mean(self.vector_array[i], axis=0)

                    vector_dir = np.diag(np.sign(avg_of_vecs))

                    # 这里只是视觉缩放，不代表物理单位真的除以 3。
                    vector_mag = np.linalg.norm(avg_of_vecs) / 3

                    s = np.diag([1.0, 1.0, vector_mag, 1.0])

                    cone_trans = np.eye(4)
                    cone_trans[:3, 3] = [0.0, 0.0, 0.12 * vector_mag]

                    cyl_mat = r3d.trans_and_rot(dyn.pos, vector_dir @ dyn.rot) @ s

                    cone_mat = r3d.trans_and_rot(dyn.pos, vector_dir @ dyn.rot) @ cone_trans

                    self.vec_cyl_transforms[i].set_transform_and_color(cyl_mat, QUAD_COLOR[i % len(QUAD_COLOR)] + (1.0,))
                    self.vec_cone_transforms[i].set_transform_and_color(cone_mat, QUAD_COLOR[i % len(QUAD_COLOR)] + (1.0,))

                if self.vis_acc_arrows:
                    if len(self.vector_array[i]) > 10:
                        self.vector_array[i].pop(0)

                    self.vector_array[i].append(dyn.acc)

                    avg_of_vecs = np.mean(self.vector_array[i], axis=0)

                    vector_dir = np.diag(np.sign(avg_of_vecs))

                    vector_mag = np.linalg.norm(avg_of_vecs) / 3

                    s = np.diag([1.0, 1.0, vector_mag, 1.0])

                    cone_trans = np.eye(4)
                    cone_trans[:3, 3] = [0.0, 0.0, 0.12 * vector_mag]

                    cyl_mat = r3d.trans_and_rot(dyn.pos, vector_dir @ dyn.rot) @ s

                    cone_mat = r3d.trans_and_rot(dyn.pos, vector_dir @ dyn.rot) @ cone_trans

                    self.vec_cyl_transforms[i].set_transform_and_color(cyl_mat, QUAD_COLOR[i % len(QUAD_COLOR)] + (1.0,))
                    self.vec_cone_transforms[i].set_transform_and_color(cone_mat, QUAD_COLOR[i % len(QUAD_COLOR)] + (1.0,))

                matrix = r3d.translate(dyn.pos)
                if collisions['drone'][i] > 0.0 or collisions['ground'][i] > 0.0 or collisions['obstacle'][i] > 0.0:
                    # RGB 三个通道分别编码 drone/obstacle/ground 三类碰撞，透明度固定为 0.4。
                    self.collision_transforms[i].set_transform_and_color(matrix, (
                        (collisions['drone'][i] > 0.0) * 1.0, (collisions['obstacle'][i] > 0.0) * 1.0,
                        (collisions['ground'][i] > 0.0) * 1.0, 0.4))
                else:
                    self.collision_transforms[i].set_transform_and_color(matrix, (0, 0, 0, 0.0))

    # 多机渲染入口。
    # `human` 模式额外处理窗口摆放和键盘交互；`rgb_array` 模式则纯离屏输出录像帧。
    def render_chase(self, all_dynamics, goals, collisions, mode='human', obstacles=None, first_spawn=None):
        import gym_art.quadrotor_multi.rendering3d as r3d

        if mode == 'human':
            if self.window_target is None:

                self.window_target = r3d.WindowTarget(self.window_w, self.window_h, resizable=self.resizable)
                if first_spawn is None:
                    first_spawn = self.window_target.location()

                newx = first_spawn[0]+((self.scene_index % 3) * self.window_w)
                newy = first_spawn[1]+((self.scene_index // 3) * self.window_h)

                self.window_target.set_location(newx, newy)

                if self.viewpoint == 'global':
                   self.window_target.draw_axes()

                self.keys = self.pygl_window.key.KeyStateHandler()
                self.window_target.window.push_handlers(self.keys)
                self.window_target.window.on_key_release = self.window_on_key_release
                self._make_scene()

            self.window_smooth_change_view()
            self.update_state(all_dynamics=all_dynamics, goals=goals, obstacles=obstacles, collisions=collisions)
            self.cam3p.look_at(*self.chase_cam.look_at())
            r3d.draw(self.scene, self.cam3p, self.window_target)
            return None, first_spawn
        elif mode == 'rgb_array':
            if self.video_target is None:
                self.video_target = r3d.FBOTarget(self.window_h, self.window_h)
                self._make_scene()
            self.update_state(all_dynamics=all_dynamics, goals=goals, obstacles=obstacles, collisions=collisions)
            self.cam3p.look_at(*self.chase_cam.look_at())
            r3d.draw(self.scene, self.cam3p, self.video_target)
            return np.flipud(self.video_target.read()), None

    # 键盘驱动的实时镜头切换逻辑。
    # 支持在 global/local 间切换、绑定到不同 drone，以及旋转/缩放/平移辅助视角。
    def window_smooth_change_view(self):
        if len(self.keys) == 0:
            return

        key = self.pygl_window.key

        symbol = list(self.keys)
        if key.NUM_0 <= symbol[0] <= key.NUM_9:
            index = min(symbol[0] - key.NUM_0, self.num_agents - 1)
            self.camera_drone_index = index
            self.viewpoint = 'local'
            self.chase_cam = ChaseCamera(view_dist=self.diameter * 15)
            self.chase_cam.reset(self.goals[index][0:3], self.dynamics[index].pos, self.dynamics[index].vel)
            return

        if self.keys[key.L]:
            self.viewpoint = 'local'
            self.chase_cam = ChaseCamera(view_dist=self.diameter * 15)
            self.chase_cam.reset(self.goals[0][0:3], self.dynamics[0].pos, self.dynamics[0].vel)
            return
        if self.keys[key.G]:
            self.viewpoint = 'global'
            self.chase_cam = GlobalCamera(view_dist=2.5)
            goal = np.mean(self.goals, axis=0)
            self.chase_cam.reset(view_dist=2.5, center=goal)

        # if not isinstance(self.chase_cam, GlobalCamera):
        #     return

        if self.keys[key.LEFT]:
            # <- Left Rotation :
            self.chase_cam.phi -= self.camera_rot_step_size
        if self.keys[key.RIGHT]:
            # -> Right Rotation :
            self.chase_cam.phi += self.camera_rot_step_size
        if self.keys[key.UP]:
            self.chase_cam.theta -= self.camera_rot_step_size
        if self.keys[key.DOWN]:
            self.chase_cam.theta += self.camera_rot_step_size
        if self.keys[key.Z]:
            # Zoom In
            self.chase_cam.radius -= self.camera_zoom_step_size
        if self.keys[key.X]:
            # Zoom Out
            self.chase_cam.radius += self.camera_zoom_step_size
        if self.keys[key.Q]:
            # Decrease the step size of Rotation
            if self.camera_rot_step_size <= np.pi / 18:
                print('Current rotation step size for camera is the minimum!')
            else:
                self.camera_rot_step_size /= 2
        if self.keys[key.P]:
            # Increase the step size of Rotation
            if self.camera_rot_step_size >= np.pi / 2:
                print('Current rotation step size for camera is the maximum!')
            else:
                self.camera_rot_step_size *= 2
        if self.keys[key.W]:
            # Decrease the step size of Zoom
            if self.camera_zoom_step_size <= 0.1:
                print('Current zoom step size for camera is the minimum!')
            else:
                self.camera_zoom_step_size -= 0.1
        if self.keys[key.O]:
            # Increase the step size of Zoom
            if self.camera_zoom_step_size >= 2.0:
                print('Current zoom step size for camera is the maximum!')
            else:
                self.camera_zoom_step_size += 0.1
        if self.keys[key.J]:
            self.chase_cam.center += np.array([0., 0., self.camera_mov_step_size])
        if self.keys[key.N]:
            self.chase_cam.center += np.array([0., 0., -self.camera_mov_step_size])
        if self.keys[key.B]:
            angle = self.chase_cam.phi + np.pi / 2
            move_step = np.array([np.cos(angle), np.sin(angle), 0]) * self.camera_mov_step_size
            self.chase_cam.center -= move_step
        if self.keys[key.M]:
            angle = self.chase_cam.phi + np.pi / 2
            move_step = np.array([np.cos(angle), np.sin(angle), 0]) * self.camera_mov_step_size
            self.chase_cam.center += move_step

        if self.keys[key.NUM_ADD]:
            self.formation_size += 0.1
        elif self.keys[key.NUM_SUBTRACT]:
            self.formation_size -= 0.1

    # `window_on_key_release` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def window_on_key_release(self, symbol, modifiers):
        key = self.pygl_window.key

        self.keys = key.KeyStateHandler()
        self.window_target.window.push_handlers(self.keys)
