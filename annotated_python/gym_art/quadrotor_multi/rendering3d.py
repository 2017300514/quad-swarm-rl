# 中文注释副本；原始文件：gym_art/quadrotor_multi/rendering3d.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件属于多机四旋翼仿真环境的一部分，负责环境状态、物理过程或配套工具中的某一环。
# 它的上游通常来自场景配置、动力学状态或训练动作，下游会流向观测构造、奖励结算、碰撞处理或可视化。

# 下面开始文件或代码块自带的文档字符串；如果源码作者已经解释设计意图，应优先结合它理解上下文。
"""
3D rendering framework
"""
# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from __future__ import division
from copy import deepcopy
import os
import six
import sys
import itertools
import noise
import ctypes

if "Apple" in sys.version:
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from gymnasium import error

print('IMPORTING OPENGL RENDERING MODULE. THIS SHOULD NOT BE IMPORTED IN HEADLESS MODE!')

try:
    # 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
    import pyglet
    pyglet.options['debug_gl'] = False
except ImportError as e:
        raise ImportError('''
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    ''')
    # reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

try:
    # 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
    from pyglet.gl import *
except ImportError as e:
        raise ImportError('''
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    ''')
    # reraise(prefix="Error occured while running `from pyglet.gl import *`",suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
import math
import numpy as np

# `get_display` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def get_display(spec):
    # 下面的文档字符串通常由源码作者提供，用来补充模块职责、输入输出约束或使用方式。
    """Convert a display specification (such as :0) into an actual Display
    object.

    pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return None
    elif isinstance(spec, six.string_types):
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))

# TODO can we get some of this from Pyglet?
# `FBOTarget` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class FBOTarget(object):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, width, height):

        shape = (width, height, 3)
        self.shape = shape

        self.fbo = GLuint(0)
        glGenFramebuffers(1, ctypes.byref(self.fbo))
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # renderbuffer for depth
        self.depth = GLuint(0)
        glGenRenderbuffers(1, ctypes.byref(self.depth))
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, *shape)
        # ??? (from songho.ca/opengl/gl_fbo.html)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
            GL_RENDERBUFFER, self.depth)

        # texture for RGB
        self.tex = GLuint(0)
        glGenTextures(1, ctypes.byref(self.tex))
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, shape[0], shape[1], 0,
            GL_RGB, GL_UNSIGNED_BYTE, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, self.tex, 0)

        # test - ok to comment out?
        draw_buffers = (GLenum * 1)(GL_COLOR_ATTACHMENT0)
        glDrawBuffers(1, draw_buffers)
        # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

        self.fb_array = np.zeros(shape, dtype=np.uint8)

    # `bind` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        draw_buffers = (GLenum * 1)(GL_COLOR_ATTACHMENT0)
        glDrawBuffers(1, draw_buffers)
        glViewport(0, 0, *self.shape[:2])

    # `finish` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def finish(self):
        glReadPixels(0, 0, self.shape[1], self.shape[0],
            GL_RGB, GL_UNSIGNED_BYTE, self.fb_array.ctypes.data)

    # `read` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def read(self):
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return self.fb_array


# `WindowTarget` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class WindowTarget(object):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, width, height, display=None, resizable=True):

        is_travis = 'TRAVIS' in os.environ
        if is_travis:
            # quick hack to hopefully fix a crash in Travis CI
            config = Config(double_buffer=True, depth_size=16)
        else:
            antialiasing_x = 4
            config = Config(double_buffer=True, depth_size=16, sample_buffers=1, samples=antialiasing_x)

        display = get_display(display)
        # vsync is set to false to speed up FBO-only renders, we enable before draw
        self.window = pyglet.window.Window(display=display,
            width=width, height=height, resizable=resizable, # style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS,
            visible=True, vsync=False, config=config
        )
        self.window.on_close = self.close
        self.shape = (width, height, 3)
        # `on_resize` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
        def on_resize(w, h):
            self.shape = (w, h, 3)
        if resizable:
            self.window.on_resize = on_resize

    # `close` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def close(self):
        self.window.close()

    # `bind` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def bind(self):
        self.window.switch_to()
        # self.window.set_vsync(True)
        self.window.dispatch_events()
        # fix for retina displays
        viewportw, viewporth = self.window.get_viewport_size()
        if viewportw > self.window.width:
            glViewport(0, 0, viewportw, viewporth)
        else:
            glViewport(0, 0, self.window.width, self.window.height)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    # `location` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def location(self):
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return self.window.get_location()

    # `set_location` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def set_location(self, x, y):
        self.window.set_location(x, y)

    # `draw_axes` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def draw_axes(self):
        # define the axes vertices and colors
        axes = pyglet.graphics.vertex_list(3,
                                           ('v3f', (0, 0, 0, 600, 0, 0, 0, 600, 0)),
                                           ('c3B', (255, 0, 0, 255, 0, 0, 0, 255, 0)))

        # draw the axes
        axes.draw(pyglet.gl.GL_LINES)

    # `finish` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def finish(self):
        self.window.flip()
        # self.window.set_vsync(False)


# `Camera` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class Camera(object):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, fov):
        self.fov = fov
        self.lookat = None

    # `look_at` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def look_at(self, eye, target, up):
        self.lookat = (eye, target, up)

    # TODO other ways to set the view matrix

    # private
    # `_matrix` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def _matrix(self, shape):
        aspect = float(shape[0]) / shape[1]
        znear = 0.1
        zfar = 100.0
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, aspect, znear, zfar)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # will make sense once more than one way of setting view matrix
        # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
        assert sum([x is not None for x in (self.lookat,)]) < 2

        if self.lookat is not None:
            eye, target, up = (list(x) for x in self.lookat)
            gluLookAt(*(eye + target + up))


# TODO we can add user-controlled lighting, etc. to this
# `Scene` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class Scene(object):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, batches, bgcolor=(0,0,0)):
        self.batches = batches
        self.bgcolor = bgcolor

        # [-1] == 0 means it's a directional light
        self.lights = [np.array([np.cos(t), np.sin(t), 0.0, 0.0])
            for t in 0.2 + np.linspace(0, 2*np.pi, 4)[:-1]]

    # call only once GL context is ready
    # `initialize` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def initialize(self):
        glShadeModel(GL_SMOOTH)
        glEnable(GL_LIGHTING)

        #glFogi(GL_FOG_MODE, GL_LINEAR)
        #glFogf(GL_FOG_START, 20.0) # Fog Start Depth
        #glFogf(GL_FOG_END, 100.0) # Fog End Depth
        #glEnable(GL_FOG)

        amb, diff, spec = (1.0 / len(self.lights)) * np.array([0.4, 1.2, 0.5])
        for i, light in enumerate(self.lights):
            # TODO fix lights in world space instead of camera space
            glLightfv(GL_LIGHT0 + i, GL_POSITION, (GLfloat * 4)(*light))
            glLightfv(GL_LIGHT0 + i, GL_AMBIENT, (GLfloat * 4)(amb, amb, amb, 1))
            glLightfv(GL_LIGHT0 + i, GL_DIFFUSE, (GLfloat * 4)(diff, diff, diff, 1))
            glLightfv(GL_LIGHT0 + i, GL_SPECULAR, (GLfloat * 4)(spec, spec, spec, 1))
            glEnable(GL_LIGHT0 + i)


# `draw` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def draw(scene, camera, target):

    target.bind() # sets viewport

    r, g, b = scene.bgcolor
    glClearColor(r, g, b, 1.0)
    glFrontFace(GL_CCW)
    glCullFace(GL_BACK)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    camera._matrix(target.shape)
    #view = (GLfloat * 16)()
    #glGetFloatv(GL_MODELVIEW_MATRIX, view)
    #view = np.array(view).reshape((4,4)).T

    for batch in scene.batches:
        batch.draw()

    target.finish()


# `SceneNode` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class SceneNode(object):
    # `_build_children` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def _build_children(self, batch):
        # hack - should go somewhere else
        if not isinstance(self.children, type([])):
            self.children = [self.children]
        for c in self.children:
            c.build(batch, self.pyg_grp)

    # default impl
    # `collide_sphere` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def collide_sphere(self, x, radius):
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return any(c.collide_sphere(x, radius) for c in self.children)

# `World` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class World(SceneNode):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, children):
        self.children = children
        self.pyg_grp = None

    # `build` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def build(self, batch):
        #batch.add(3, GL_LINES, None, ('v2f', (0, 0, 5, 0, 0, 5)))
        self._build_children(batch)

# `Transform` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class Transform(SceneNode):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, transform, children):
        self.t = transform
        self.mat_inv = np.linalg.inv(transform)
        self.children = children

    # `build` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def build(self, batch, parent):
        self.pyg_grp = _PygTransform(self.t, parent=parent)
        self._build_children(batch)
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return self.pyg_grp

    # `set_transform` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def set_transform(self, t):
        self.pyg_grp.set_matrix(t)
        self.mat_inv = np.linalg.inv(t)

    # `set_transform_nocollide` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def set_transform_nocollide(self, t):
        self.pyg_grp.set_matrix(t)

    # `set_transform_and_color` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def set_transform_and_color(self, t, rgba):
        self.pyg_grp.set_matrix(t)
        for child in self.children:
            child.set_rgba(rgba[0], rgba[1], rgba[2], rgba[3])

    # `collide_sphere` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def collide_sphere(self, x, radius):
        xh = [x[0], x[1], x[2], 1]
        xlocal = np.matmul(self.mat_inv, xh)[:3]
        rlocal = radius * self.mat_inv[0,0]
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return any(c.collide_sphere(xlocal, rlocal) for c in self.children)

# `BackToFront` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class BackToFront(SceneNode):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, children):
        self.children = children

    # `build` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def build(self, batch, parent):
        self.pyg_grp = pyglet.graphics.Group(parent=parent)
        for i, c in enumerate(self.children):
            ordering = pyglet.graphics.OrderedGroup(i, parent=self.pyg_grp)
            c.build(batch, ordering)
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return self.pyg_grp

# `Color` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class Color(SceneNode):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, color, children):
        self.color = color
        self.children = children

    # `build` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def build(self, batch, parent):
        self.pyg_grp = _PygColor(self.color, parent=parent)
        self._build_children(batch)
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return self.pyg_grp

    # `set_rgb` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def set_rgb(self, r, g, b):
        self.pyg_grp.set_rgb(r, g, b)

    # `set_rgba` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def set_rgba(self, r, g, b, a):
        self.pyg_grp.set_rgba(r, g, b, a)

# `transform_and_color` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def transform_and_color(transform, color, children):
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return Transform(transform, Color(color, children))

# `transform_and_dual_color` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def transform_and_dual_color(transform, color_1, color_2, children_1, children_2):
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return Transform(transform, [Color(color_1, children_1), Color(color_2, children_2)])

TEX_CHECKER = 0
TEX_XOR = 1
TEX_NOISE_GAUSSIAN = 2
TEX_NOISE_PERLIN = 3
TEX_OILY = 4
TEX_VORONOI = 5

# `random_textype` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def random_textype():
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return np.random.randint(TEX_VORONOI + 1)

# `ProceduralTexture` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class ProceduralTexture(SceneNode):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, style, scale, children):
        self.children = children
        # linear is default, those w/ nearest must overwrite
        self.mag_filter = GL_LINEAR
        if style == TEX_CHECKER:
            image = np.zeros((256, 256))
            image[:128,:128] = 1.0
            image[128:,128:] = 1.0
            self.mag_filter = GL_NEAREST
        elif style == TEX_XOR:
            x, y = np.meshgrid(range(256), range(256))
            image = np.float32(np.bitwise_xor(np.uint8(x), np.uint8(y)))
            self.mag_filter = GL_NEAREST
        elif style == TEX_NOISE_GAUSSIAN:
            nz = np.random.normal(size=(256,256))
            # 这里按 observation space 上下界裁剪邻居观测，避免极端数值破坏网络训练时的输入尺度。
            image = np.clip(nz, -3, 3)
        elif style == TEX_NOISE_PERLIN:
            t = np.linspace(0, 1, 256)
            nzfun = lambda x, y: noise.pnoise2(x, y,
                octaves=10, persistence=0.8, repeatx=1, repeaty=1)
            image = np.vectorize(nzfun)(*np.meshgrid(t, t))
        elif style == TEX_OILY:
            # from upvector.com "Intro to Procedural Textures"
            t = np.linspace(0, 4, 256)
            nzfun = lambda x, y: noise.snoise2(x, y,
                octaves=10, persistence=0.45, repeatx=4, repeaty=4)
            nz = np.vectorize(nzfun)(*np.meshgrid(t, t))

            t = np.linspace(0, 20*np.pi, 257)[:-1]
            x, y = np.meshgrid(t, t)
            image = np.sin(x + 8*nz)
        elif style == TEX_VORONOI:
            npts = 64
            points = np.random.uniform(size=(npts, 2))
            # make it tile
            shifts = itertools.product([-1, 0, 1], [-1, 0, 1])
            points = np.vstack([points + shift for shift in shifts])
            unlikely = np.any(np.logical_or(points < -0.25, points > 1.25), axis=1)
            points = np.delete(points, np.where(unlikely), axis=0)
            a = np.full((256, 256), np.inf)
            t = np.linspace(0, 1, 256)
            x, y = np.meshgrid(t, t)
            for p in points:
                dist2 = (x - p[0])**2 + (y - p[1])**2
                a = np.minimum(a, dist2)
            image = np.sqrt(a)
        else:
            raise KeyError("style does not exist")

        low, high = 255.0 * scale[0], 255.0 * scale[1]
        _scale_to_inplace(image, low, high)
        self.tex = _np2tex(image)

    # `build` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def build(self, batch, parent):
        self.pyg_grp = _PygTexture(tex=self.tex,
            mag_filter=self.mag_filter, parent=parent)
        self._build_children(batch)
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return self.pyg_grp


#
# these functions return 4x4 rotation matrix suitable to construct Transform
# or to mutate Transform via set_matrix
#
# `scale` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def scale(s):
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return np.diag([s, s, s, 1.0])

# `translate` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def translate(x):
    r = np.eye(4)
    r[:3,3] = x
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return r

# `trans_and_rot` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def trans_and_rot(t, r):
    m = np.eye(4)
    m[:3,:3] = r
    m[:3,3] = t
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return m

# `rotz` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def rotz(theta):
    r = np.eye(4)
    r[:2,:2] = _rot2d(theta)
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return r

# `roty` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def roty(theta):
    r = np.eye(4)
    r2d = _rot2d(theta)
    r[[0,0,2,2],[0,2,0,2]] = _rot2d(theta).flatten()
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return r

# `rotx` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def rotx(theta):
    r = np.eye(4)
    r[1:3,1:3] = _rot2d(theta)
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return r

# `_PygTransform` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class _PygTransform(pyglet.graphics.Group):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, transform=np.eye(4), parent=None):
        super().__init__(parent)
        self.set_matrix(transform)

    # `set_matrix` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def set_matrix(self, transform):
        # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
        assert transform.shape == (4, 4)
        # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
        assert np.all(transform[3,:] == [0, 0, 0, 1])
        self.matrix_raw = (GLfloat * 16)(*transform.T.flatten())

    # `set_state` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def set_state(self):
        glPushMatrix()
        glMultMatrixf(self.matrix_raw)

    # `unset_state` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def unset_state(self):
        glPopMatrix()

# `_PygColor` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class _PygColor(pyglet.graphics.Group):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, color, parent=None):
        super().__init__(parent)
        if len(color) == 3:
            self.set_rgb(*color)
        else:
            self.set_rgba(*color)

    # `set_rgb` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def set_rgb(self, r, g, b):
        self.set_rgba(r, g, b, 1.0)

    # `set_rgba` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def set_rgba(self, r, g, b, a):
        self.dcolor = (GLfloat * 4)(r, g, b, a)
        spec_whiteness = 0.8
        r, g, b = (1.0 - spec_whiteness) * np.array([r, g, b]) + spec_whiteness
        self.scolor = (GLfloat * 4)(r, g, b, a)

    # `set_state` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def set_state(self):
        if self.dcolor[-1] < 1.0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.dcolor)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, self.scolor)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, (GLfloat)(8.0))

    # `unset_state` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def unset_state(self):
        if self.dcolor[-1] < 1.0:
            glDisable(GL_BLEND)

# `_PygTexture` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class _PygTexture(pyglet.graphics.Group):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, tex, mag_filter, parent=None):
        super().__init__(parent=parent)

        self.tex = tex
        glBindTexture(GL_TEXTURE_2D, self.tex.id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

        # anisotropic texturing helps a lot with checkerboard floors
        anisotropy = (GLfloat)()
        glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy)

    # `set_state` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def set_state(self):
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (GLfloat * 4)(1,1,1,1))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(1,1,1,1))
        glEnable(self.tex.target)
        glBindTexture(self.tex.target, self.tex.id)

    # `unset_state` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def unset_state(self):
        glDisable(self.tex.target)


# `_PygAlphaBlending` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class _PygAlphaBlending(pyglet.graphics.Group):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, parent=None):
        super().__init__(parent=parent)

    # `set_state` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def set_state(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # `unset_state` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def unset_state(self):
        glDisable(GL_BLEND)

Batch = pyglet.graphics.Batch

# we only implement collision detection between primitives and spheres.
# the world-coordinate sphere is transformed according to the scene graph
# into the primitive's canonical coordinate system.
# this simplifies the math a lot, although it might be less efficient
# than directly testing the sphere against the transformed primitive
# in world coordinates.
# `SphereCollision` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class SphereCollision(object):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, radius):
        self.radius = radius
    # `collide_sphere` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def collide_sphere(self, x, radius):
        c = np.sum(x ** 2) < (self.radius + radius) ** 2
        if c: print("collided with sphere")
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return c

# `AxisBoxCollision` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class AxisBoxCollision(object):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, corner0, corner1):
        self.corner0, self.corner1 = corner0, corner1
    # `collide_sphere` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def collide_sphere(self, x, radius):
        nearest = np.maximum(self.corner0, np.minimum(x, self.corner1))
        c = np.sum((x - nearest)**2) < radius**2
        if c: print("collided with box")
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return c

# `CapsuleCollision` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class CapsuleCollision(object):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, radius, height):
        self.radius, self.height = radius, height
    # `collide_sphere` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def collide_sphere(self, x, radius):
        z = min(max(0, x[2]), self.height)
        nearest = [0, 0, z]
        c = np.sum((x - nearest)**2) < (self.radius + radius)**2
        if c: print("collided with capsule")
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return c

#
# these are the 3d primitives that can be added to a pyglet.graphics.Batch.
# construct them with the shape functions below.
#
# `BatchElement` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class BatchElement(SceneNode):
    # `build` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def build(self, batch, parent):
        self.batch_args[2] = parent
        batch.add(*self.batch_args)

    # `collide_sphere` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def collide_sphere(self, x, radius):
        if self.collider is not None:
            # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
            return self.collider.collide_sphere(x, radius)
        else:
            # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
            return False

# `Mesh` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class Mesh(BatchElement):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, verts, normals=None, st=None, collider=None):
        if len(verts.shape) != 2 or verts.shape[1] != 3:
            raise ValueError('verts must be an N x 3 NumPy array')

        N = verts.shape[0]
        # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
        assert int(N) % 3 == 0

        if st is not None:
            # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
            assert st.shape == (N, 2)

        if normals is None:
            # compute normals implied by triangle faces
            normals = deepcopy(verts)

            for i in range(0, N, 3):
                v0, v1, v2 = verts[i:(i+3),:]
                d0, d1 = (v1 - v0), (v2 - v1)
                n = _normalize(np.cross(d0, d1))
                normals[i:(i+3),:] = n

        self.batch_args = [N, pyglet.gl.GL_TRIANGLES, None,
            ('v3f/static', list(verts.flatten())),
            ('n3f/static', list(normals.flatten())),
        ]
        if st is not None:
            self.batch_args.append(('t2f/static', list(st.flatten())))
        self.collider = collider

# `TriStrip` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class TriStrip(BatchElement):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, verts, normals, collider=None):
        N, dim = verts.shape
        # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
        assert dim == 3
        # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
        assert normals.shape == verts.shape

        self.batch_args = [N, pyglet.gl.GL_TRIANGLE_STRIP, None,
            ('v3f/static', list(verts.flatten())),
            ('n3f/static', list(normals.flatten()))
        ]
        self.collider = collider

# `TriFan` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class TriFan(BatchElement):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, verts, normals, collider=None):
        N, dim = verts.shape
        # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
        assert dim == 3
        # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
        assert normals.shape == verts.shape

        self.batch_args = [N, pyglet.gl.GL_TRIANGLE_FAN, None,
            ('v3f/static', list(verts.flatten())),
            ('n3f/static', list(normals.flatten()))
        ]
        self.collider = collider

# a box centered on the origin
# `box` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def box(x, y, z):
    corner1 = np.array([x,y,z]) / 2
    corner0 = -corner1
    v = box_mesh(x, y, z)
    collider = AxisBoxCollision(corner0, corner1)
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return Mesh(v, collider=collider)

# Box representing the environment. Shortcut way to add walls during visualization
# `envBox` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def envBox(x, y, z):
    # corner1 = np.array([-x, -y, z])
    # corner2 = np.array([x, y, 0])
    corner1 = np.array([x/2, y/2, z])
    corner2 = -corner1
    corner2[2] = 0
    v = room_mesh(x, y, z)
    collider = AxisBoxCollision(corner1, corner2)
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return Mesh(v, collider=collider)

# cylinder sitting on xy plane pointing +z
# `cylinder` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def cylinder(radius, height, sections):
    v, n = cylinder_strip(radius, height, sections)
    collider = CapsuleCollision(radius, height)
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return TriStrip(v, n, collider=collider)

# cylinder sitting on xy plane pointing +x
# `rod` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def rod(radius, length, sections):
    v, n = cylinder_strip(radius, length, sections)
    collider = CapsuleCollision(radius, length)
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return TriStrip(v, n, collider=collider)

# cone sitting on xy plane pointing +z
# `cone` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def cone(radius, height, sections):
    # TODO collision detectoin
    v, n = cone_strip(radius, height, sections)
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return TriStrip(v, n)

# arrow sitting on xy plane pointing +z
# `arrow` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def arrow(radius, height, sections):
    v, n = arrow_strip(radius, height, sections)
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return TriStrip(v, n)

# sphere centered on origin, n tris will be about TODO * facets
# `sphere` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def sphere(radius, facets, facet_range=None):
    v, n = sphere_strip(radius, facets, facet_range)
    collider = SphereCollision(radius)
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return TriStrip(v, n, collider=collider)

# square in xy plane centered on origin
# dim: (w, h)
# srange, trange: desired min/max (s, t) tex coords
# `rect` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def rect(dim, srange=(0,1), trange=(0,1)):
    v = np.array([
        [1, 1, 0], [-1, 1, 0], [1, -1, 0],
        [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])
    v = np.matmul(v, np.diag([dim[0] / 2.0, dim[1] / 2.0, 0]))
    n = _withz(0 * v, 1)
    s0, s1 = srange
    t0, t1 = trange
    st = np.array([
        [s1, t1], [s0, t1], [s1, t0],
        [s0, t1], [s0, t0], [s1, t0]])
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return Mesh(v, n, st)


# `circle` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def circle(radius, facets):
    v, n = circle_fan(radius, facets)
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return TriFan(v, n)

#
# low-level primitive builders. return vertex/normal/texcoord arrays.
# good if you want to apply transforms directly to the points, etc.
#

# box centered on origin with given dimensions.
# no normals, but Mesh ctor will estimate them perfectly
# `box_mesh` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def box_mesh(x, y, z):
    vtop = np.array([[x, y, z], [x, -y, z], [-x, -y, z], [-x, y, z]])
    vbottom = deepcopy(vtop)
    vbottom[:,2] = -vbottom[:,2]
    # 这里执行观测拼接，把分散的物理特征重组为策略网络期望的固定顺序向量。
    v = 0.5 * np.concatenate([vtop, vbottom], axis=0)
    t = np.array([[1, 3, 2,], [1, 4, 3,], [1, 2, 5,], [2, 6, 5,], [2, 3, 6,], [3, 7, 6,], [3, 4, 8,], [3, 8, 7,], [4, 1, 8,], [1, 5, 8,], [5, 6, 7,], [5, 7, 8,]]) - 1
    t = t.flatten()
    v = v[t,:]
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return v

# need a different mesh function for the envBox
# `room_mesh` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def room_mesh(x, y, z):
    vtop = np.array([[x/2, y/2, z], [x/2, -y/2, z], [-x/2, -y/2, z], [-x/2, y/2, z]])
    vbottom = deepcopy(vtop)
    vbottom[:, 2] = -1.0
    # 这里执行观测拼接，把分散的物理特征重组为策略网络期望的固定顺序向量。
    v = np.concatenate([vtop, vbottom], axis=0)
    # triangle meshes
    t = np.array(
        [[1, 2, 3, ], [1, 3, 4, ], [1, 5, 2, ], [2, 5, 6, ], [2, 6, 3, ], [3, 6, 7, ], [3, 8, 4, ], [3, 7, 8, ],
         [4, 8, 1, ], [1, 8, 5, ], [5, 7, 6, ], [5, 8, 7, ]]) - 1
    t = t.flatten()
    v = v[t, :]
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return v

# circle in the x-y plane
# `circle_fan` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def circle_fan(radius, sections):
    t = np.linspace(0, 2 * np.pi, sections + 1)[:,None]
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    v = np.hstack([x, y, 0*t])
    v = np.vstack([[0, 0, 0], v])
    n = _withz(0 * v, 1)
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return v, n

# cylinder sitting on the x-y plane
# `cylinder_strip` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def cylinder_strip(radius, height, sections):
    t = np.linspace(0, 2 * np.pi, sections + 1)[:,None]
    x = radius * np.cos(t)
    y = radius * np.sin(t)

    base = np.hstack([x, y, 0*t])
    top = np.hstack([x, y, height + 0*t])
    strip_sides = _to_strip(np.hstack([base[:,None,:], top[:,None,:]]))
    normals_sides = _withz(strip_sides / radius, 0)

    # `make_cap` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def make_cap(circle, normal_z):
        height = circle[0,2]
        center = _withz(0 * circle, height)
        if normal_z > 0:
            strip = _to_strip(np.hstack([circle[:,None,:], center[:,None,:]]))
        else:
            strip = _to_strip(np.hstack([center[:,None,:], circle[:,None,:]]))
        normals = _withz(0 * strip, normal_z)
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return strip, normals

    vbase, nbase = make_cap(base, -1)
    vtop, ntop = make_cap(top, 1)
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return (
        np.vstack([strip_sides, vbase, vtop]),
        np.vstack([normals_sides, nbase, ntop]))

# cone sitting on the x-y plane
# `cone_strip` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def cone_strip(radius, height, sections):
    t = np.linspace(0, 2 * np.pi, sections + 1)[:,None]
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    base = np.hstack([x, y, 0*t])

    top = _withz(0 * base, height)
    vside = _to_strip(np.hstack([base[:,None,:], top[:,None,:]]))
    base_tangent = np.cross(_npa(0, 0, 1), base)
    top_to_base = base - top
    normals = _normalize(np.cross(top_to_base, base_tangent))
    nside = _to_strip(np.hstack([normals[:,None,:], normals[:,None,:]]))

    base_ctr = 0 * base
    vbase = _to_strip(np.hstack([base_ctr[:,None,:], base[:,None,:]]))
    nbase = _withz(0 * vbase, -1)

    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return np.vstack([vside, vbase]), np.vstack([nside, nbase])

# sphere centered on origin
# `sphere_strip` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def sphere_strip(radius, resolution, resolution_range=None):
    t = np.linspace(-1, 1, resolution)
    u, v = np.meshgrid(t, t)
    vtx = []
    panel = np.zeros((resolution, resolution, 3))
    inds = list(range(3))
    res_range = (0, resolution-1) if not resolution_range else resolution_range
    for i in range(3):
        panel[:,:,inds[0]] = u
        panel[:,:,inds[1]] = v
        panel[:,:,inds[2]] = 1
        norms = np.linalg.norm(panel, axis=2)
        panel = panel / norms[:,:,None]
        for _ in range(2):
            for j in range(res_range[0], res_range[1]):
                strip = deepcopy(panel[[j,j+1],:,:].transpose([1,0,2]).reshape((-1,3)))
                degen0 = deepcopy(strip[0,:])
                degen1 = deepcopy(strip[-1,:])
                vtx.extend([degen0, strip, degen1])
            panel *= -1
            panel = np.flip(panel, axis=1)
        inds = [inds[-1]] + inds[:-1]

    n = np.vstack(vtx)
    v = radius * n
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return v, n

# arrow sitting on x-y plane
# `arrow_strip` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def arrow_strip(radius, height, facets):
    cyl_r = radius
    cyl_h = 0.75 * height
    cone_h = height - cyl_h
    cone_half_angle = np.radians(30)
    cone_r = 1.5 * cone_h * np.tan(cone_half_angle)
    vcyl, ncyl = cylinder_strip(cyl_r, cyl_h, facets)
    vcone, ncone = cone_strip(cone_r, cone_h, facets)
    vcone[:,2] += cyl_h
    v = np.vstack([vcyl, vcone])
    n = np.vstack([ncyl, ncone])
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return v, n


#
# private helper functions, not part of API
#
# `_npa` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def _npa(*args):
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return np.array(args)

# `_normalize` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def _normalize(x):
    if len(x.shape) == 1:
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return x / np.linalg.norm(x)
    elif len(x.shape) == 2:
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return x / np.linalg.norm(x, axis=1)[:,None]
    else:
        # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
        assert False

# `_withz` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def _withz(a, z):
    b = 0 + a
    b[:,2] = z
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return b

# `_rot2d` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def _rot2d(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return np.array([[c, -s], [s, c]])

# add degenerate tris, convert from N x 2 x 3 to 2N+2 x 3
# `_to_strip` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def _to_strip(strip):
    s0 = strip[0,0,:]
    s1 = strip[-1,-1,:]
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return np.vstack([s0, np.reshape(strip, (-1, 3)), s1])

# `_scale_to_inplace` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def _scale_to_inplace(a, min1, max1):
    id0 = id(a)
    min0 = np.min(a.flatten())
    max0 = np.max(a.flatten())
    scl = (max1 - min1) / (max0 - min0)
    shift = - (scl * min0) + min1
    a *= scl
    a += shift
    # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
    assert id(a) == id0

# `_np2tex` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def _np2tex(a):
    # TODO color
    w, h = a.shape
    b = np.uint8(a).tobytes()
    # 这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。
    assert len(b) == w * h
    img = pyglet.image.ImageData(w, h, "L", b)
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return img.get_texture()


