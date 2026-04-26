# 中文注释副本；原始文件：gym_art/quadrotor_multi/quadrotor_multi_visualization.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import copy

# 导入当前模块依赖。
import pyglet

# 导入当前模块依赖。
from gym_art.quadrotor_multi.quad_utils import *
from gym_art.quadrotor_multi.quadrotor_visualization import ChaseCamera, SideCamera, quadrotor_simple_3dmodel, \
    # 执行这一行逻辑。
    quadrotor_3dmodel


# Global Camera
# 定义类 `GlobalCamera`。
class GlobalCamera(object):
    # 定义函数 `__init__`。
    def __init__(self, view_dist=2.0):
        # 保存或更新 `radius` 的值。
        self.radius = view_dist
        # 保存或更新 `theta` 的值。
        self.theta = np.pi / 2
        # 保存或更新 `phi` 的值。
        self.phi = 0.0
        # 保存或更新 `center` 的值。
        self.center = np.array([0., 0., 2.])

    # 定义函数 `reset`。
    def reset(self, view_dist=2.0, center=np.array([0., 0., 2.])):
        # 保存或更新 `center` 的值。
        self.center = center

    # 定义函数 `step`。
    def step(self, center=np.array([0., 0., 2.])):
        # 当前代码块暂时不执行实际逻辑。
        pass

    # 定义函数 `look_at`。
    def look_at(self):
        # 保存或更新 `up` 的值。
        up = npa(0, 0, 1)
        # 保存或更新 `center` 的值。
        center = self.center  # pattern center
        # 保存或更新 `eye` 的值。
        eye = center + self.radius * np.array(
            [np.sin(self.theta) * np.cos(self.phi), np.sin(self.theta) * np.sin(self.phi), np.cos(self.theta)])
        # 返回当前函数的结果。
        return eye, center, up

# 定义类 `TopDownCamera`。
class TopDownCamera(object):
    # 定义函数 `__init__`。
    def __init__(self, view_dist=3.0):
        # 保存或更新 `radius` 的值。
        self.radius = view_dist
        # 保存或更新 `theta` 的值。
        self.theta = np.pi / 2
        # 保存或更新 `phi` 的值。
        self.phi = 0.0
        # 保存或更新 `center` 的值。
        self.center = np.array([0., 0., 15.])

    # 定义函数 `reset`。
    def reset(self, view_dist=2.0, center=np.array([0., 0., 5.])):
        # 保存或更新 `center` 的值。
        self.center = np.array([0., 0., 15.])
        #self.center = center

    # 定义函数 `step`。
    def step(self, center=np.array([0., 0., 2.])):
        # 当前代码块暂时不执行实际逻辑。
        pass

    # 定义函数 `look_at`。
    def look_at(self):
        # 保存或更新 `up` 的值。
        up = npa(0, 1, 0)
        # 保存或更新 `eye` 的值。
        eye = self.center  # pattern center
        # 保存或更新 `center` 的值。
        center = self.center - np.array([0, 0, 2])
        # 保存或更新 `center` 的值。
        center = (center/np.linalg.norm(center)) * self.radius
        # 返回当前函数的结果。
        return eye, center, up

# 定义类 `CornerCamera`。
class CornerCamera(object):
    # 定义函数 `__init__`。
    def __init__(self, view_dist=4.0, room_dims=np.array([10, 10, 10]), corner_index=0):
        # 保存或更新 `radius` 的值。
        self.radius = view_dist
        # 保存或更新 `theta` 的值。
        self.theta = np.pi / 2
        # 保存或更新 `phi` 的值。
        self.phi = 0.0
        # 保存或更新 `center` 的值。
        self.center = np.array([0., 0., 2.])
        # 保存或更新 `corner_index` 的值。
        self.corner_index = corner_index
        # 保存或更新 `room_dims` 的值。
        self.room_dims = room_dims
        # 根据条件决定是否进入当前分支。
        if corner_index == 0:
            # 保存或更新 `center` 的值。
            self.center = np.array([-self.room_dims[0] / 2, -self.room_dims[1] / 2, self.room_dims[2]])
        # 当上一分支不满足时，继续判断新的条件。
        elif corner_index == 1:
            # 保存或更新 `center` 的值。
            self.center = np.array([self.room_dims[0] / 2, -self.room_dims[1] / 2, self.room_dims[2]])
        # 当上一分支不满足时，继续判断新的条件。
        elif corner_index == 2:
            # 保存或更新 `center` 的值。
            self.center = np.array([-self.room_dims[0] / 2, self.room_dims[1] / 2, self.room_dims[2]])
        # 当上一分支不满足时，继续判断新的条件。
        elif corner_index == 3:
            # 保存或更新 `center` 的值。
            self.center = np.array([self.room_dims[0] / 2, self.room_dims[1] / 2, self.room_dims[2]])

    # 定义函数 `reset`。
    def reset(self, view_dist=4.0, center=None):
        # 根据条件决定是否进入当前分支。
        if center is not None:
            # 保存或更新 `center` 的值。
            self.center = center
        # 当上一分支不满足时，继续判断新的条件。
        elif self.corner_index == 0:
            # 保存或更新 `center` 的值。
            self.center = np.array([-self.room_dims[0] / 2, -self.room_dims[1] / 2, self.room_dims[2]])
        # 当上一分支不满足时，继续判断新的条件。
        elif self.corner_index == 1:
            # 保存或更新 `center` 的值。
            self.center = np.array([self.room_dims[0] / 2, -self.room_dims[1] / 2, self.room_dims[2]])
        # 当上一分支不满足时，继续判断新的条件。
        elif self.corner_index == 2:
            # 保存或更新 `center` 的值。
            self.center = np.array([-self.room_dims[0] / 2, self.room_dims[1] / 2, self.room_dims[2]])
        # 当上一分支不满足时，继续判断新的条件。
        elif self.corner_index == 3:
            # 保存或更新 `center` 的值。
            self.center = np.array([self.room_dims[0] / 2, self.room_dims[1] / 2, self.room_dims[2]])

    # 定义函数 `step`。
    def step(self, center=np.array([0., 0., 2.])):
        # 当前代码块暂时不执行实际逻辑。
        pass

    # 定义函数 `look_at`。
    def look_at(self):
        # 保存或更新 `up` 的值。
        up = npa(0, 0, 1)
        # 保存或更新 `eye` 的值。
        eye = self.center  # pattern center
        # 保存或更新 `center` 的值。
        center = self.center - np.array([0, 0, 2])
        # 保存或更新 `center` 的值。
        center = (center/np.linalg.norm(center)) * self.radius
        # 返回当前函数的结果。
        return eye, center, up

# 定义类 `TopDownFollowCamera`。
class TopDownFollowCamera(object):
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
        # 保存或更新 `pos_smooth` 的值。
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos

    # return eye, center, up suitable for gluLookAt
    # 定义函数 `look_at`。
    def look_at(self):
        # 保存或更新 `up` 的值。
        up = npa(0, 1, 0)
        # 保存或更新 `eye` 的值。
        eye = self.pos_smooth + np.array([0, 0, 5])
        # 保存或更新 `center` 的值。
        center = self.pos_smooth
        # 返回当前函数的结果。
        return eye, center, up


# 定义类 `Quadrotor3DSceneMulti`。
class Quadrotor3DSceneMulti:
    # 保存或更新 `first_spawn_x` 的值。
    first_spawn_x = 0

    # 定义函数 `__init__`。
    def __init__(
            self, w, h,
            quad_arm=None, models=None, walls_visible=True, resizable=True, goal_diameter=None,
            viewpoint='chase', obs_hw=None, room_dims=(10, 10, 10), num_agents=8, obstacles=None,
            render_speed=1.0, formation_size=-1.0, vis_vel_arrows=True, vis_acc_arrows=True, viz_traces=100, viz_trace_nth_step=1,
            num_obstacles=0, scene_index=0
    # 这里开始一个新的代码块。
    ):
        # 保存或更新 `pygl_window` 的值。
        self.pygl_window = __import__('pyglet.window', fromlist=['key'])
        # 保存或更新 `keys` 的值。
        self.keys = None  # keypress handler, initialized later

        # 根据条件决定是否进入当前分支。
        if obs_hw is None:
            # 保存或更新 `obs_hw` 的值。
            obs_hw = [64, 64]

        # 保存或更新 `window_target` 的值。
        self.window_target = None
        # 同时更新 `window_w`, `window_h` 等变量。
        self.window_w, self.window_h = w, h
        # 保存或更新 `resizable` 的值。
        self.resizable = resizable
        # 保存或更新 `viewpoint` 的值。
        self.viewpoint = viewpoint
        # 保存或更新 `obs_hw` 的值。
        self.obs_hw = copy.deepcopy(obs_hw)
        # 保存或更新 `walls_visible` 的值。
        self.walls_visible = walls_visible
        # 保存或更新 `scene_index` 的值。
        self.scene_index = scene_index

        # 保存或更新 `quad_arm` 的值。
        self.quad_arm = quad_arm
        # 保存或更新 `models` 的值。
        self.models = models
        # 保存或更新 `room_dims` 的值。
        self.room_dims = room_dims

        # 同时更新 `quad_transforms`, `shadow_transforms`, `goal_transforms` 等变量。
        self.quad_transforms, self.shadow_transforms, self.goal_transforms = [], [], []

        # 根据条件决定是否进入当前分支。
        if goal_diameter:
            # 保存或更新 `goal_forced_diameter` 的值。
            self.goal_forced_diameter = goal_diameter
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `goal_forced_diameter` 的值。
            self.goal_forced_diameter = None

        # 保存或更新 `diameter` 的值。
        self.diameter = self.goal_diameter = -1
        # 调用 `update_goal_diameter` 执行当前处理。
        self.update_goal_diameter()

        # 根据条件决定是否进入当前分支。
        if self.viewpoint == 'chase':
            # 保存或更新 `chase_cam` 的值。
            self.chase_cam = ChaseCamera(view_dist=self.diameter * 15)
        # 当上一分支不满足时，继续判断新的条件。
        elif self.viewpoint == 'side':
            # 保存或更新 `chase_cam` 的值。
            self.chase_cam = SideCamera(view_dist=self.diameter * 15)
        # 当上一分支不满足时，继续判断新的条件。
        elif self.viewpoint == 'global':
            # 保存或更新 `chase_cam` 的值。
            self.chase_cam = GlobalCamera(view_dist=2.5)
        # 当上一分支不满足时，继续判断新的条件。
        elif self.viewpoint == 'topdown':
            # 保存或更新 `chase_cam` 的值。
            self.chase_cam = TopDownCamera(view_dist=2.5)
        # 当上一分支不满足时，继续判断新的条件。
        elif self.viewpoint == 'topdownfollow':
            # 保存或更新 `chase_cam` 的值。
            self.chase_cam = TopDownFollowCamera(view_dist=2.5)
        # 当上一分支不满足时，继续判断新的条件。
        elif self.viewpoint[:-1] == 'corner':
            # 保存或更新 `chase_cam` 的值。
            self.chase_cam = CornerCamera(view_dist=4.0, room_dims=self.room_dims, corner_index=int(self.viewpoint[-1]))

        # 保存或更新 `fpv_lookat` 的值。
        self.fpv_lookat = None

        # 保存或更新 `scene` 的值。
        self.scene = None
        # 保存或更新 `window_target` 的值。
        self.window_target = None
        # 保存或更新 `obs_target` 的值。
        self.obs_target = None
        # 保存或更新 `video_target` 的值。
        self.video_target = None

        # 保存或更新 `obstacles` 的值。
        self.obstacles = None
        # 根据条件决定是否进入当前分支。
        if obstacles:
            # 保存或更新 `obstacles` 的值。
            self.obstacles = obstacles

        # Save parameters to help transfer from global camera to local camera
        # 保存或更新 `goals` 的值。
        self.goals = None
        # 保存或更新 `dynamics` 的值。
        self.dynamics = None
        # 保存或更新 `num_agents` 的值。
        self.num_agents = num_agents
        # 保存或更新 `camera_drone_index` 的值。
        self.camera_drone_index = 0

        # Aux camera moving
        # 保存或更新 `standard_render_speed` 的值。
        standard_render_speed = 1.0
        # 保存或更新 `speed_ratio` 的值。
        speed_ratio = render_speed / standard_render_speed
        # 保存或更新 `camera_rot_step_size` 的值。
        self.camera_rot_step_size = np.pi / 45 * speed_ratio
        # 保存或更新 `camera_zoom_step_size` 的值。
        self.camera_zoom_step_size = 0.1 * speed_ratio
        # 保存或更新 `camera_mov_step_size` 的值。
        self.camera_mov_step_size = 0.1 * speed_ratio
        # 保存或更新 `formation_size` 的值。
        self.formation_size = formation_size
        # 保存或更新 `vis_vel_arrows` 的值。
        self.vis_vel_arrows = vis_vel_arrows
        # 保存或更新 `vis_acc_arrows` 的值。
        self.vis_acc_arrows = vis_acc_arrows
        # 保存或更新 `viz_traces` 的值。
        self.viz_traces = 50
        # 保存或更新 `viz_trace_nth_step` 的值。
        self.viz_trace_nth_step = viz_trace_nth_step
        # 保存或更新 `vector_array` 的值。
        self.vector_array = [[] for _ in range(num_agents)]
        # 保存或更新 `store_path_every_n` 的值。
        self.store_path_every_n = 1
        # 保存或更新 `store_path_count` 的值。
        self.store_path_count = 0
        # 保存或更新 `path_store` 的值。
        self.path_store = [[] for _ in range(num_agents)]

    # 定义函数 `update_goal_diameter`。
    def update_goal_diameter(self):
        # 根据条件决定是否进入当前分支。
        if self.quad_arm is not None:
            # 保存或更新 `diameter` 的值。
            self.diameter = self.quad_arm
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `diameter` 的值。
            self.diameter = np.linalg.norm(self.models[0].params['motor_pos']['xyz'][:2])

        # 根据条件决定是否进入当前分支。
        if self.goal_forced_diameter:
            # 保存或更新 `goal_diameter` 的值。
            self.goal_diameter = self.goal_forced_diameter
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `goal_diameter` 的值。
            self.goal_diameter = self.diameter

    # 定义函数 `update_env`。
    def update_env(self, room_dims):
        # 保存或更新 `room_dims` 的值。
        self.room_dims = room_dims
        # 调用 `_make_scene` 执行当前处理。
        self._make_scene()

    # 定义函数 `_make_scene`。
    def _make_scene(self):
        # 导入当前模块依赖。
        import gym_art.quadrotor_multi.rendering3d as r3d

        # 保存或更新 `cam1p` 的值。
        self.cam1p = r3d.Camera(fov=90.0)
        # 保存或更新 `cam3p` 的值。
        self.cam3p = r3d.Camera(fov=45.0)

        # 同时更新 `quad_transforms`, `shadow_transforms`, `goal_transforms`, `collision_transforms` 等变量。
        self.quad_transforms, self.shadow_transforms, self.goal_transforms, self.collision_transforms = [], [], [], []
        # 同时更新 `obstacle_transforms`, `vec_cyl_transforms`, `vec_cone_transforms` 等变量。
        self.obstacle_transforms, self.vec_cyl_transforms, self.vec_cone_transforms = [], [], []
        # 保存或更新 `path_transforms` 的值。
        self.path_transforms = [[] for _ in range(self.num_agents)]

        # 保存或更新 `shadow_circle` 的值。
        shadow_circle = r3d.circle(0.75 * self.diameter, 32)
        # 保存或更新 `collision_sphere` 的值。
        collision_sphere = r3d.sphere(0.75 * self.diameter, 32)

        # 保存或更新 `arrow_cylinder` 的值。
        arrow_cylinder = r3d.cylinder(0.005, 0.12, 16)
        # 保存或更新 `arrow_cone` 的值。
        arrow_cone = r3d.cone(0.01, 0.04, 16)
        # 保存或更新 `path_sphere` 的值。
        path_sphere = r3d.sphere(0.15 * self.diameter, 16)

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i, model in enumerate(self.models):
            # 根据条件决定是否进入当前分支。
            if model is not None:
                # 保存或更新 `quad_transform` 的值。
                quad_transform = quadrotor_3dmodel(model, quad_id=i)
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # 保存或更新 `quad_transform` 的值。
                quad_transform = quadrotor_simple_3dmodel(self.diameter)
            # 调用 `append` 执行当前处理。
            self.quad_transforms.append(quad_transform)

            # 调用 `append` 执行当前处理。
            self.shadow_transforms.append(
                r3d.transform_and_color(np.eye(4), (0, 0, 0, 0.0), shadow_circle)
            )
            # 调用 `append` 执行当前处理。
            self.collision_transforms.append(
                r3d.transform_and_color(np.eye(4), (0, 0, 0, 0.0), collision_sphere)
            )
            # 根据条件决定是否进入当前分支。
            if self.vis_vel_arrows:
                # 调用 `append` 执行当前处理。
                self.vec_cyl_transforms.append(
                    r3d.transform_and_color(np.eye(4), (1, 1, 1), arrow_cylinder)
                )
                # 调用 `append` 执行当前处理。
                self.vec_cone_transforms.append(
                    r3d.transform_and_color(np.eye(4), (1, 1, 1), arrow_cone)
                )
            # 根据条件决定是否进入当前分支。
            if self.vis_acc_arrows:
                # 调用 `append` 执行当前处理。
                self.vec_cyl_transforms.append(
                    r3d.transform_and_color(np.eye(4), (1, 1, 1), arrow_cylinder)
                )
                # 调用 `append` 执行当前处理。
                self.vec_cone_transforms.append(
                    r3d.transform_and_color(np.eye(4), (1, 1, 1), arrow_cone)
                )

            # 根据条件决定是否进入当前分支。
            if self.viz_traces:
                # 保存或更新 `color` 的值。
                color = QUAD_COLOR[i % len(QUAD_COLOR)] + (1.0,)
                # 遍历当前序列或迭代器，逐项执行下面的逻辑。
                for j in range(self.viz_traces):
                    # 执行这一行逻辑。
                    self.path_transforms[i].append(r3d.transform_and_color(np.eye(4), color, path_sphere))

        # TODO make floor size or walls to indicate world_box
        # 保存或更新 `floor` 的值。
        floor = r3d.ProceduralTexture(2, (0.85, 0.95),
                                      r3d.rect((100, 100), (0, 100), (0, 100)))
        # 调用 `update_goal_diameter` 执行当前处理。
        self.update_goal_diameter()
        # 保存或更新 `chase_cam.view_dist` 的值。
        self.chase_cam.view_dist = self.diameter * 15

        # 调用 `create_goals` 执行当前处理。
        self.create_goals()

        # 保存或更新 `bodies` 的值。
        bodies = [r3d.BackToFront([floor, st]) for st in self.shadow_transforms]
        # 调用 `extend` 执行当前处理。
        bodies.extend(self.goal_transforms)
        # 调用 `extend` 执行当前处理。
        bodies.extend(self.quad_transforms)
        # 调用 `extend` 执行当前处理。
        bodies.extend(self.vec_cyl_transforms)
        # 调用 `extend` 执行当前处理。
        bodies.extend(self.vec_cone_transforms)
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for path in self.path_transforms:
            # 调用 `extend` 执行当前处理。
            bodies.extend(path)
        # visualize walls of the room if True
        # 根据条件决定是否进入当前分支。
        if self.walls_visible:
            # 保存或更新 `room` 的值。
            room = r3d.ProceduralTexture(r3d.random_textype(), (0.75, 0.85), r3d.envBox(*self.room_dims))
            # 调用 `append` 执行当前处理。
            bodies.append(room)

        # 根据条件决定是否进入当前分支。
        if self.obstacles:
            # 调用 `create_obstacles` 执行当前处理。
            self.create_obstacles()
            # 调用 `extend` 执行当前处理。
            bodies.extend(self.obstacle_transforms)

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

        # Collision spheres have to be added in the ending after everything has been rendered, as it transparent
        # 保存或更新 `bodies` 的值。
        bodies = []
        # 调用 `extend` 执行当前处理。
        bodies.extend(self.collision_transforms)
        # 保存或更新 `world` 的值。
        world = r3d.World(bodies)
        # 保存或更新 `batch` 的值。
        batch = r3d.Batch()
        # 调用 `build` 执行当前处理。
        world.build(batch)
        # 调用 `extend` 执行当前处理。
        self.scene.batches.extend([batch])

    # 定义函数 `create_obstacles`。
    def create_obstacles(self):
        # 导入当前模块依赖。
        import gym_art.quadrotor_multi.rendering3d as r3d
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for item in self.obstacles.pos_arr:
            # 保存或更新 `color` 的值。
            color = OBST_COLOR_3
            # 保存或更新 `obst_height` 的值。
            obst_height = self.room_dims[2]
            # 保存或更新 `obstacle_transform` 的值。
            obstacle_transform = r3d.transform_and_color(np.eye(4), color, r3d.cylinder(
                radius=self.obstacles.size / 2.0, height=obst_height, sections=64))

            # 调用 `append` 执行当前处理。
            self.obstacle_transforms.append(obstacle_transform)

    # 定义函数 `update_obstacles`。
    def update_obstacles(self, obstacles):
        # 导入当前模块依赖。
        import gym_art.quadrotor_multi.rendering3d as r3d

        # 根据条件决定是否进入当前分支。
        if len(obstacles.pos_arr) == 1:
            # 返回当前函数的结果。
            return

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i, g in enumerate(obstacles.pos_arr):
            # self.obstacle_transforms[i].set_transform(r3d.translate(g.pos))
            # 保存或更新 `pos_update` 的值。
            pos_update = [g[0], g[1], g[2] - self.room_dims[2] / 2]

            # color = QUAD_COLOR
            # 执行这一行逻辑。
            self.obstacle_transforms[i].set_transform_and_color(r3d.translate(pos_update), OBST_COLOR_4)

    #def create_arrows(self, envs):
    #
    #    self.

    # 定义函数 `create_goals`。
    def create_goals(self):
        # 导入当前模块依赖。
        import gym_art.quadrotor_multi.rendering3d as r3d

        # 保存或更新 `goal_sphere` 的值。
        goal_sphere = r3d.sphere(0.1 / 2, 18)
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(len(self.models)):
            # 保存或更新 `color` 的值。
            color = QUAD_COLOR[i % len(QUAD_COLOR)]
            # 保存或更新 `goal_transform` 的值。
            goal_transform = r3d.transform_and_color(np.eye(4), color, goal_sphere)
            # 调用 `append` 执行当前处理。
            self.goal_transforms.append(goal_transform)

    # 定义函数 `update_goals`。
    def update_goals(self, goals):
        # 导入当前模块依赖。
        import gym_art.quadrotor_multi.rendering3d as r3d

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i, g in enumerate(goals):
            # 执行这一行逻辑。
            self.goal_transforms[i].set_transform(r3d.translate(g[0:3]))

    # 定义函数 `update_models`。
    def update_models(self, models):
        # 保存或更新 `models` 的值。
        self.models = models

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

    # 定义函数 `reset`。
    def reset(self, goals, dynamics, obstacles, collisions):
        # 保存或更新 `goals` 的值。
        self.goals = goals
        # 保存或更新 `dynamics` 的值。
        self.dynamics = dynamics
        # 保存或更新 `vector_array` 的值。
        self.vector_array = [[] for _ in range(self.num_agents)]
        # 保存或更新 `path_store` 的值。
        self.path_store = [[] for _ in range(self.num_agents)]

        # 根据条件决定是否进入当前分支。
        if self.viewpoint == 'global':
            # 保存或更新 `goal` 的值。
            goal = np.mean(goals, axis=0)
            # 保存或更新 `chase_cam.reset(view_dist` 的值。
            self.chase_cam.reset(view_dist=2.5, center=goal)
        # 当上一分支不满足时，继续判断新的条件。
        elif self.viewpoint[:-1] == 'corner' or self.viewpoint == 'topdown':
            # 调用 `reset` 执行当前处理。
            self.chase_cam.reset()
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `goal` 的值。
            goal = goals[self.camera_drone_index]  # TODO: make a camera that can look at all drones
            # 调用 `reset` 执行当前处理。
            self.chase_cam.reset(goal[0:3], dynamics[self.camera_drone_index].pos,
                                 dynamics[self.camera_drone_index].vel)

        # 调用 `update_state` 执行当前处理。
        self.update_state(dynamics, goals, obstacles, collisions)

    # 定义函数 `update_state`。
    def update_state(self, all_dynamics, goals, obstacles, collisions):
        # 导入当前模块依赖。
        import gym_art.quadrotor_multi.rendering3d as r3d

        # 根据条件决定是否进入当前分支。
        if self.scene:
            # 根据条件决定是否进入当前分支。
            if self.viewpoint == 'global' or self.viewpoint[:-1] == 'corner' or self.viewpoint == 'topdown':
                # 保存或更新 `goal` 的值。
                goal = np.mean(goals, axis=0)
                # 保存或更新 `chase_cam.step(center` 的值。
                self.chase_cam.step(center=goal)
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # 调用 `step` 执行当前处理。
                self.chase_cam.step(all_dynamics[self.camera_drone_index].pos,
                                    all_dynamics[self.camera_drone_index].vel)
                # 保存或更新 `fpv_lookat` 的值。
                self.fpv_lookat = all_dynamics[self.camera_drone_index].look_at()
            # use this to get trails on the goals and visualize the paths they follow
            # bodies = []
            # bodies.extend(self.goal_transforms)
            # world = r3d.World(bodies)
            # batch = r3d.Batch()
            # world.build(batch)
            # self.scene.batches.extend([batch])
            # 保存或更新 `store_path_count` 的值。
            self.store_path_count += 1
            # 保存或更新 `update_goals(goals` 的值。
            self.update_goals(goals=goals)
            # 根据条件决定是否进入当前分支。
            if self.obstacles:
                # 调用 `update_obstacles` 执行当前处理。
                self.update_obstacles(obstacles)

            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i, dyn in enumerate(all_dynamics):
                # 保存或更新 `matrix` 的值。
                matrix = r3d.trans_and_rot(dyn.pos, dyn.rot)
                # 执行这一行逻辑。
                self.quad_transforms[i].set_transform_nocollide(matrix)

                # 保存或更新 `translation` 的值。
                translation = r3d.translate(dyn.pos)

                # 根据条件决定是否进入当前分支。
                if self.viz_traces and self.store_path_count % self.viz_trace_nth_step == 0:
                    # 根据条件决定是否进入当前分支。
                    if len(self.path_store[i]) >= self.viz_traces:
                        # 执行这一行逻辑。
                        self.path_store[i].pop(0)

                    # 执行这一行逻辑。
                    self.path_store[i].append(translation)
                    # 保存或更新 `color_rgba` 的值。
                    color_rgba = QUAD_COLOR[i % len(QUAD_COLOR)] + (1.0,)
                    # 保存或更新 `path_storage_length` 的值。
                    path_storage_length = len(self.path_store[i])
                    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
                    for k in range(path_storage_length):
                        # 保存或更新 `scale` 的值。
                        scale = k / path_storage_length + 0.01
                        # 保存或更新 `transformation` 的值。
                        transformation = self.path_store[i][k] @ r3d.scale(scale)
                        # 执行这一行逻辑。
                        self.path_transforms[i][k].set_transform_and_color(transformation, color_rgba)

                # shadow_pos = 0 + dyn.pos
                # shadow_pos[2] = 0.001  # avoid z-fighting
                # matrix = r3d.translate(shadow_pos)
                # self.shadow_transforms[i].set_transform_nocollide(matrix)

                # 根据条件决定是否进入当前分支。
                if self.vis_vel_arrows:
                    # 根据条件决定是否进入当前分支。
                    if len(self.vector_array[i]) > 10:
                        # 执行这一行逻辑。
                        self.vector_array[i].pop(0)

                    # 执行这一行逻辑。
                    self.vector_array[i].append(dyn.vel)

                    # Get average of the vectors
                    # 保存或更新 `avg_of_vecs` 的值。
                    avg_of_vecs = np.mean(self.vector_array[i], axis=0)

                    # Calculate direction
                    # 保存或更新 `vector_dir` 的值。
                    vector_dir = np.diag(np.sign(avg_of_vecs))

                    # Calculate magnitude and divide by 3 (for aesthetics)
                    # 保存或更新 `vector_mag` 的值。
                    vector_mag = np.linalg.norm(avg_of_vecs) / 3

                    # 保存或更新 `s` 的值。
                    s = np.diag([1.0, 1.0, vector_mag, 1.0])

                    # 保存或更新 `cone_trans` 的值。
                    cone_trans = np.eye(4)
                    # 保存或更新 `cone_trans[:3, 3]` 的值。
                    cone_trans[:3, 3] = [0.0, 0.0, 0.12 * vector_mag]

                    # 保存或更新 `cyl_mat` 的值。
                    cyl_mat = r3d.trans_and_rot(dyn.pos, vector_dir @ dyn.rot) @ s

                    # 保存或更新 `cone_mat` 的值。
                    cone_mat = r3d.trans_and_rot(dyn.pos, vector_dir @ dyn.rot) @ cone_trans

                    # 执行这一行逻辑。
                    self.vec_cyl_transforms[i].set_transform_and_color(cyl_mat, QUAD_COLOR[i % len(QUAD_COLOR)] + (1.0,))
                    # 执行这一行逻辑。
                    self.vec_cone_transforms[i].set_transform_and_color(cone_mat, QUAD_COLOR[i % len(QUAD_COLOR)] + (1.0,))

                # 根据条件决定是否进入当前分支。
                if self.vis_acc_arrows:
                    # 根据条件决定是否进入当前分支。
                    if len(self.vector_array[i]) > 10:
                        # 执行这一行逻辑。
                        self.vector_array[i].pop(0)

                    # 执行这一行逻辑。
                    self.vector_array[i].append(dyn.acc)

                    # Get average of the vectors
                    # 保存或更新 `avg_of_vecs` 的值。
                    avg_of_vecs = np.mean(self.vector_array[i], axis=0)

                    # Calculate direction
                    # 保存或更新 `vector_dir` 的值。
                    vector_dir = np.diag(np.sign(avg_of_vecs))

                    # Calculate magnitude and divide by 3 (for aesthetics)
                    # 保存或更新 `vector_mag` 的值。
                    vector_mag = np.linalg.norm(avg_of_vecs) / 3

                    # 保存或更新 `s` 的值。
                    s = np.diag([1.0, 1.0, vector_mag, 1.0])

                    # 保存或更新 `cone_trans` 的值。
                    cone_trans = np.eye(4)
                    # 保存或更新 `cone_trans[:3, 3]` 的值。
                    cone_trans[:3, 3] = [0.0, 0.0, 0.12 * vector_mag]

                    # 保存或更新 `cyl_mat` 的值。
                    cyl_mat = r3d.trans_and_rot(dyn.pos, vector_dir @ dyn.rot) @ s

                    # 保存或更新 `cone_mat` 的值。
                    cone_mat = r3d.trans_and_rot(dyn.pos, vector_dir @ dyn.rot) @ cone_trans

                    # 执行这一行逻辑。
                    self.vec_cyl_transforms[i].set_transform_and_color(cyl_mat, QUAD_COLOR[i % len(QUAD_COLOR)] + (1.0,))
                    # 执行这一行逻辑。
                    self.vec_cone_transforms[i].set_transform_and_color(cone_mat, QUAD_COLOR[i % len(QUAD_COLOR)] + (1.0,))

                # 保存或更新 `matrix` 的值。
                matrix = r3d.translate(dyn.pos)
                # 根据条件决定是否进入当前分支。
                if collisions['drone'][i] > 0.0 or collisions['ground'][i] > 0.0 or collisions['obstacle'][i] > 0.0:
                    # Multiplying by 1 converts bool into float
                    # 执行这一行逻辑。
                    self.collision_transforms[i].set_transform_and_color(matrix, (
                        (collisions['drone'][i] > 0.0) * 1.0, (collisions['obstacle'][i] > 0.0) * 1.0,
                        (collisions['ground'][i] > 0.0) * 1.0, 0.4))
                # 当前置条件都不满足时，执行兜底分支。
                else:
                    # 执行这一行逻辑。
                    self.collision_transforms[i].set_transform_and_color(matrix, (0, 0, 0, 0.0))

    # 定义函数 `render_chase`。
    def render_chase(self, all_dynamics, goals, collisions, mode='human', obstacles=None, first_spawn=None):
        # 导入当前模块依赖。
        import gym_art.quadrotor_multi.rendering3d as r3d

        # 根据条件决定是否进入当前分支。
        if mode == 'human':
            # 根据条件决定是否进入当前分支。
            if self.window_target is None:

                # 保存或更新 `window_target` 的值。
                self.window_target = r3d.WindowTarget(self.window_w, self.window_h, resizable=self.resizable)
                # 根据条件决定是否进入当前分支。
                if first_spawn is None:
                    # 保存或更新 `first_spawn` 的值。
                    first_spawn = self.window_target.location()

                # 保存或更新 `newx` 的值。
                newx = first_spawn[0]+((self.scene_index % 3) * self.window_w)
                # 保存或更新 `newy` 的值。
                newy = first_spawn[1]+((self.scene_index // 3) * self.window_h)

                # 调用 `set_location` 执行当前处理。
                self.window_target.set_location(newx, newy)

                # 根据条件决定是否进入当前分支。
                if self.viewpoint == 'global':
                   # 调用 `draw_axes` 执行当前处理。
                   self.window_target.draw_axes()

                # 保存或更新 `keys` 的值。
                self.keys = self.pygl_window.key.KeyStateHandler()
                # 调用 `push_handlers` 执行当前处理。
                self.window_target.window.push_handlers(self.keys)
                # 保存或更新 `window_target.window.on_key_release` 的值。
                self.window_target.window.on_key_release = self.window_on_key_release
                # 调用 `_make_scene` 执行当前处理。
                self._make_scene()

            # 调用 `window_smooth_change_view` 执行当前处理。
            self.window_smooth_change_view()
            # 保存或更新 `update_state(all_dynamics` 的值。
            self.update_state(all_dynamics=all_dynamics, goals=goals, obstacles=obstacles, collisions=collisions)
            # 调用 `look_at` 执行当前处理。
            self.cam3p.look_at(*self.chase_cam.look_at())
            # 调用 `draw` 执行当前处理。
            r3d.draw(self.scene, self.cam3p, self.window_target)
            # 返回当前函数的结果。
            return None, first_spawn
        # 当上一分支不满足时，继续判断新的条件。
        elif mode == 'rgb_array':
            # 根据条件决定是否进入当前分支。
            if self.video_target is None:
                # 保存或更新 `video_target` 的值。
                self.video_target = r3d.FBOTarget(self.window_h, self.window_h)
                # 调用 `_make_scene` 执行当前处理。
                self._make_scene()
            # 保存或更新 `update_state(all_dynamics` 的值。
            self.update_state(all_dynamics=all_dynamics, goals=goals, obstacles=obstacles, collisions=collisions)
            # 调用 `look_at` 执行当前处理。
            self.cam3p.look_at(*self.chase_cam.look_at())
            # 调用 `draw` 执行当前处理。
            r3d.draw(self.scene, self.cam3p, self.video_target)
            # 返回当前函数的结果。
            return np.flipud(self.video_target.read()), None

    # 定义函数 `window_smooth_change_view`。
    def window_smooth_change_view(self):
        # 根据条件决定是否进入当前分支。
        if len(self.keys) == 0:
            # 返回当前函数的结果。
            return

        # 保存或更新 `key` 的值。
        key = self.pygl_window.key

        # 保存或更新 `symbol` 的值。
        symbol = list(self.keys)
        # 根据条件决定是否进入当前分支。
        if key.NUM_0 <= symbol[0] <= key.NUM_9:
            # 保存或更新 `index` 的值。
            index = min(symbol[0] - key.NUM_0, self.num_agents - 1)
            # 保存或更新 `camera_drone_index` 的值。
            self.camera_drone_index = index
            # 保存或更新 `viewpoint` 的值。
            self.viewpoint = 'local'
            # 保存或更新 `chase_cam` 的值。
            self.chase_cam = ChaseCamera(view_dist=self.diameter * 15)
            # 调用 `reset` 执行当前处理。
            self.chase_cam.reset(self.goals[index][0:3], self.dynamics[index].pos, self.dynamics[index].vel)
            # 返回当前函数的结果。
            return

        # 根据条件决定是否进入当前分支。
        if self.keys[key.L]:
            # 保存或更新 `viewpoint` 的值。
            self.viewpoint = 'local'
            # 保存或更新 `chase_cam` 的值。
            self.chase_cam = ChaseCamera(view_dist=self.diameter * 15)
            # 调用 `reset` 执行当前处理。
            self.chase_cam.reset(self.goals[0][0:3], self.dynamics[0].pos, self.dynamics[0].vel)
            # 返回当前函数的结果。
            return
        # 根据条件决定是否进入当前分支。
        if self.keys[key.G]:
            # 保存或更新 `viewpoint` 的值。
            self.viewpoint = 'global'
            # 保存或更新 `chase_cam` 的值。
            self.chase_cam = GlobalCamera(view_dist=2.5)
            # 保存或更新 `goal` 的值。
            goal = np.mean(self.goals, axis=0)
            # 保存或更新 `chase_cam.reset(view_dist` 的值。
            self.chase_cam.reset(view_dist=2.5, center=goal)

        # if not isinstance(self.chase_cam, GlobalCamera):
        #     return

        # 根据条件决定是否进入当前分支。
        if self.keys[key.LEFT]:
            # <- Left Rotation :
            # 保存或更新 `chase_cam.phi` 的值。
            self.chase_cam.phi -= self.camera_rot_step_size
        # 根据条件决定是否进入当前分支。
        if self.keys[key.RIGHT]:
            # -> Right Rotation :
            # 保存或更新 `chase_cam.phi` 的值。
            self.chase_cam.phi += self.camera_rot_step_size
        # 根据条件决定是否进入当前分支。
        if self.keys[key.UP]:
            # 保存或更新 `chase_cam.theta` 的值。
            self.chase_cam.theta -= self.camera_rot_step_size
        # 根据条件决定是否进入当前分支。
        if self.keys[key.DOWN]:
            # 保存或更新 `chase_cam.theta` 的值。
            self.chase_cam.theta += self.camera_rot_step_size
        # 根据条件决定是否进入当前分支。
        if self.keys[key.Z]:
            # Zoom In
            # 保存或更新 `chase_cam.radius` 的值。
            self.chase_cam.radius -= self.camera_zoom_step_size
        # 根据条件决定是否进入当前分支。
        if self.keys[key.X]:
            # Zoom Out
            # 保存或更新 `chase_cam.radius` 的值。
            self.chase_cam.radius += self.camera_zoom_step_size
        # 根据条件决定是否进入当前分支。
        if self.keys[key.Q]:
            # Decrease the step size of Rotation
            # 根据条件决定是否进入当前分支。
            if self.camera_rot_step_size <= np.pi / 18:
                # 调用 `print` 执行当前处理。
                print('Current rotation step size for camera is the minimum!')
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # 保存或更新 `camera_rot_step_size` 的值。
                self.camera_rot_step_size /= 2
        # 根据条件决定是否进入当前分支。
        if self.keys[key.P]:
            # Increase the step size of Rotation
            # 根据条件决定是否进入当前分支。
            if self.camera_rot_step_size >= np.pi / 2:
                # 调用 `print` 执行当前处理。
                print('Current rotation step size for camera is the maximum!')
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # 保存或更新 `camera_rot_step_size` 的值。
                self.camera_rot_step_size *= 2
        # 根据条件决定是否进入当前分支。
        if self.keys[key.W]:
            # Decrease the step size of Zoom
            # 根据条件决定是否进入当前分支。
            if self.camera_zoom_step_size <= 0.1:
                # 调用 `print` 执行当前处理。
                print('Current zoom step size for camera is the minimum!')
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # 保存或更新 `camera_zoom_step_size` 的值。
                self.camera_zoom_step_size -= 0.1
        # 根据条件决定是否进入当前分支。
        if self.keys[key.O]:
            # Increase the step size of Zoom
            # 根据条件决定是否进入当前分支。
            if self.camera_zoom_step_size >= 2.0:
                # 调用 `print` 执行当前处理。
                print('Current zoom step size for camera is the maximum!')
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # 保存或更新 `camera_zoom_step_size` 的值。
                self.camera_zoom_step_size += 0.1
        # 根据条件决定是否进入当前分支。
        if self.keys[key.J]:
            # 保存或更新 `chase_cam.center` 的值。
            self.chase_cam.center += np.array([0., 0., self.camera_mov_step_size])
        # 根据条件决定是否进入当前分支。
        if self.keys[key.N]:
            # 保存或更新 `chase_cam.center` 的值。
            self.chase_cam.center += np.array([0., 0., -self.camera_mov_step_size])
        # 根据条件决定是否进入当前分支。
        if self.keys[key.B]:
            # 保存或更新 `angle` 的值。
            angle = self.chase_cam.phi + np.pi / 2
            # 保存或更新 `move_step` 的值。
            move_step = np.array([np.cos(angle), np.sin(angle), 0]) * self.camera_mov_step_size
            # 保存或更新 `chase_cam.center` 的值。
            self.chase_cam.center -= move_step
        # 根据条件决定是否进入当前分支。
        if self.keys[key.M]:
            # 保存或更新 `angle` 的值。
            angle = self.chase_cam.phi + np.pi / 2
            # 保存或更新 `move_step` 的值。
            move_step = np.array([np.cos(angle), np.sin(angle), 0]) * self.camera_mov_step_size
            # 保存或更新 `chase_cam.center` 的值。
            self.chase_cam.center += move_step

        # 根据条件决定是否进入当前分支。
        if self.keys[key.NUM_ADD]:
            # 保存或更新 `formation_size` 的值。
            self.formation_size += 0.1
        # 当上一分支不满足时，继续判断新的条件。
        elif self.keys[key.NUM_SUBTRACT]:
            # 保存或更新 `formation_size` 的值。
            self.formation_size -= 0.1

    # 定义函数 `window_on_key_release`。
    def window_on_key_release(self, symbol, modifiers):
        # 保存或更新 `key` 的值。
        key = self.pygl_window.key

        # 保存或更新 `keys` 的值。
        self.keys = key.KeyStateHandler()
        # 调用 `push_handlers` 执行当前处理。
        self.window_target.window.push_handlers(self.keys)
