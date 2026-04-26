# 中文注释副本；原始文件：gym_art/quadrotor_multi/rendering3d.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 下面开始文档字符串说明。
"""
3D rendering framework
"""
# 导入当前模块依赖。
from __future__ import division
from copy import deepcopy
import os
import six
import sys
import itertools
import noise
import ctypes

# 根据条件决定是否进入当前分支。
if "Apple" in sys.version:
    # 根据条件决定是否进入当前分支。
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        # 保存或更新 `os.environ[DYLD_FALLBACK_LIBRARY_PATH]` 的值。
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

# 导入当前模块依赖。
from gymnasium import error

# 调用 `print` 执行当前处理。
print('IMPORTING OPENGL RENDERING MODULE. THIS SHOULD NOT BE IMPORTED IN HEADLESS MODE!')

# 尝试执行下面的逻辑，并为异常情况做准备。
try:
    # 导入当前模块依赖。
    import pyglet
    # 保存或更新 `pyglet.options[debug_gl]` 的值。
    pyglet.options['debug_gl'] = False
# 捕获前面代码可能抛出的异常。
except ImportError as e:
        # 主动抛出异常以中止或提示错误。
        raise ImportError('''
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    ''')
    # reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

# 尝试执行下面的逻辑，并为异常情况做准备。
try:
    # 导入当前模块依赖。
    from pyglet.gl import *
# 捕获前面代码可能抛出的异常。
except ImportError as e:
        # 主动抛出异常以中止或提示错误。
        raise ImportError('''
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    ''')
    # reraise(prefix="Error occured while running `from pyglet.gl import *`",suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")

# 导入当前模块依赖。
import math
import numpy as np

# 定义函数 `get_display`。
def get_display(spec):
    # 下面的文档字符串用于说明当前模块或代码块。
    """Convert a display specification (such as :0) into an actual Display
    object.

    pyglet only supports multiple Displays on Linux.
    """
    # 根据条件决定是否进入当前分支。
    if spec is None:
        # 返回当前函数的结果。
        return None
    # 当上一分支不满足时，继续判断新的条件。
    elif isinstance(spec, six.string_types):
        # 返回当前函数的结果。
        return pyglet.canvas.Display(spec)
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 主动抛出异常以中止或提示错误。
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))

# TODO can we get some of this from Pyglet?
# 定义类 `FBOTarget`。
class FBOTarget(object):
    # 定义函数 `__init__`。
    def __init__(self, width, height):

        # 保存或更新 `shape` 的值。
        shape = (width, height, 3)
        # 保存或更新 `shape` 的值。
        self.shape = shape

        # 保存或更新 `fbo` 的值。
        self.fbo = GLuint(0)
        # 调用 `glGenFramebuffers` 执行当前处理。
        glGenFramebuffers(1, ctypes.byref(self.fbo))
        # 调用 `glBindFramebuffer` 执行当前处理。
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # renderbuffer for depth
        # 保存或更新 `depth` 的值。
        self.depth = GLuint(0)
        # 调用 `glGenRenderbuffers` 执行当前处理。
        glGenRenderbuffers(1, ctypes.byref(self.depth))
        # 调用 `glBindRenderbuffer` 执行当前处理。
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth)
        # 调用 `glRenderbufferStorage` 执行当前处理。
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, *shape)
        # ??? (from songho.ca/opengl/gl_fbo.html)
        # 调用 `glBindRenderbuffer` 执行当前处理。
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        # 调用 `glFramebufferRenderbuffer` 执行当前处理。
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
            GL_RENDERBUFFER, self.depth)

        # texture for RGB
        # 保存或更新 `tex` 的值。
        self.tex = GLuint(0)
        # 调用 `glGenTextures` 执行当前处理。
        glGenTextures(1, ctypes.byref(self.tex))
        # 调用 `glBindTexture` 执行当前处理。
        glBindTexture(GL_TEXTURE_2D, self.tex)
        # 调用 `glTexImage2D` 执行当前处理。
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, shape[0], shape[1], 0,
            GL_RGB, GL_UNSIGNED_BYTE, 0)
        # 调用 `glFramebufferTexture2D` 执行当前处理。
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, self.tex, 0)

        # test - ok to comment out?
        # 保存或更新 `draw_buffers` 的值。
        draw_buffers = (GLenum * 1)(GL_COLOR_ATTACHMENT0)
        # 调用 `glDrawBuffers` 执行当前处理。
        glDrawBuffers(1, draw_buffers)
        # 断言当前条件成立，用于保护运行假设。
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

        # 保存或更新 `fb_array` 的值。
        self.fb_array = np.zeros(shape, dtype=np.uint8)

    # 定义函数 `bind`。
    def bind(self):
        # 调用 `glBindFramebuffer` 执行当前处理。
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        # 保存或更新 `draw_buffers` 的值。
        draw_buffers = (GLenum * 1)(GL_COLOR_ATTACHMENT0)
        # 调用 `glDrawBuffers` 执行当前处理。
        glDrawBuffers(1, draw_buffers)
        # 调用 `glViewport` 执行当前处理。
        glViewport(0, 0, *self.shape[:2])

    # 定义函数 `finish`。
    def finish(self):
        # 调用 `glReadPixels` 执行当前处理。
        glReadPixels(0, 0, self.shape[1], self.shape[0],
            GL_RGB, GL_UNSIGNED_BYTE, self.fb_array.ctypes.data)

    # 定义函数 `read`。
    def read(self):
        # 返回当前函数的结果。
        return self.fb_array


# 定义类 `WindowTarget`。
class WindowTarget(object):
    # 定义函数 `__init__`。
    def __init__(self, width, height, display=None, resizable=True):

        # 保存或更新 `is_travis` 的值。
        is_travis = 'TRAVIS' in os.environ
        # 根据条件决定是否进入当前分支。
        if is_travis:
            # quick hack to hopefully fix a crash in Travis CI
            # 保存或更新 `config` 的值。
            config = Config(double_buffer=True, depth_size=16)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `antialiasing_x` 的值。
            antialiasing_x = 4
            # 保存或更新 `config` 的值。
            config = Config(double_buffer=True, depth_size=16, sample_buffers=1, samples=antialiasing_x)

        # 保存或更新 `display` 的值。
        display = get_display(display)
        # vsync is set to false to speed up FBO-only renders, we enable before draw
        # 保存或更新 `window` 的值。
        self.window = pyglet.window.Window(display=display,
            width=width, height=height, resizable=resizable, # style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS,
            visible=True, vsync=False, config=config
        )
        # 保存或更新 `window.on_close` 的值。
        self.window.on_close = self.close
        # 保存或更新 `shape` 的值。
        self.shape = (width, height, 3)
        # 定义函数 `on_resize`。
        def on_resize(w, h):
            # 保存或更新 `shape` 的值。
            self.shape = (w, h, 3)
        # 根据条件决定是否进入当前分支。
        if resizable:
            # 保存或更新 `window.on_resize` 的值。
            self.window.on_resize = on_resize

    # 定义函数 `close`。
    def close(self):
        # 调用 `close` 执行当前处理。
        self.window.close()

    # 定义函数 `bind`。
    def bind(self):
        # 调用 `switch_to` 执行当前处理。
        self.window.switch_to()
        # self.window.set_vsync(True)
        # 调用 `dispatch_events` 执行当前处理。
        self.window.dispatch_events()
        # fix for retina displays
        # 同时更新 `viewportw`, `viewporth` 等变量。
        viewportw, viewporth = self.window.get_viewport_size()
        # 根据条件决定是否进入当前分支。
        if viewportw > self.window.width:
            # 调用 `glViewport` 执行当前处理。
            glViewport(0, 0, viewportw, viewporth)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 调用 `glViewport` 执行当前处理。
            glViewport(0, 0, self.window.width, self.window.height)
        # 调用 `glBindFramebuffer` 执行当前处理。
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    # 定义函数 `location`。
    def location(self):
        # 返回当前函数的结果。
        return self.window.get_location()

    # 定义函数 `set_location`。
    def set_location(self, x, y):
        # 调用 `set_location` 执行当前处理。
        self.window.set_location(x, y)

    # 定义函数 `draw_axes`。
    def draw_axes(self):
        # define the axes vertices and colors
        # 保存或更新 `axes` 的值。
        axes = pyglet.graphics.vertex_list(3,
                                           ('v3f', (0, 0, 0, 600, 0, 0, 0, 600, 0)),
                                           ('c3B', (255, 0, 0, 255, 0, 0, 0, 255, 0)))

        # draw the axes
        # 调用 `draw` 执行当前处理。
        axes.draw(pyglet.gl.GL_LINES)

    # 定义函数 `finish`。
    def finish(self):
        # 调用 `flip` 执行当前处理。
        self.window.flip()
        # self.window.set_vsync(False)


# 定义类 `Camera`。
class Camera(object):
    # 定义函数 `__init__`。
    def __init__(self, fov):
        # 保存或更新 `fov` 的值。
        self.fov = fov
        # 保存或更新 `lookat` 的值。
        self.lookat = None

    # 定义函数 `look_at`。
    def look_at(self, eye, target, up):
        # 保存或更新 `lookat` 的值。
        self.lookat = (eye, target, up)

    # TODO other ways to set the view matrix

    # private
    # 定义函数 `_matrix`。
    def _matrix(self, shape):
        # 保存或更新 `aspect` 的值。
        aspect = float(shape[0]) / shape[1]
        # 保存或更新 `znear` 的值。
        znear = 0.1
        # 保存或更新 `zfar` 的值。
        zfar = 100.0
        # 调用 `glMatrixMode` 执行当前处理。
        glMatrixMode(GL_PROJECTION)
        # 调用 `glLoadIdentity` 执行当前处理。
        glLoadIdentity()
        # 调用 `gluPerspective` 执行当前处理。
        gluPerspective(self.fov, aspect, znear, zfar)

        # 调用 `glMatrixMode` 执行当前处理。
        glMatrixMode(GL_MODELVIEW)
        # 调用 `glLoadIdentity` 执行当前处理。
        glLoadIdentity()
        # will make sense once more than one way of setting view matrix
        # 断言当前条件成立，用于保护运行假设。
        assert sum([x is not None for x in (self.lookat,)]) < 2

        # 根据条件决定是否进入当前分支。
        if self.lookat is not None:
            # 同时更新 `eye`, `target`, `up` 等变量。
            eye, target, up = (list(x) for x in self.lookat)
            # 调用 `gluLookAt` 执行当前处理。
            gluLookAt(*(eye + target + up))


# TODO we can add user-controlled lighting, etc. to this
# 定义类 `Scene`。
class Scene(object):
    # 定义函数 `__init__`。
    def __init__(self, batches, bgcolor=(0,0,0)):
        # 保存或更新 `batches` 的值。
        self.batches = batches
        # 保存或更新 `bgcolor` 的值。
        self.bgcolor = bgcolor

        # [-1] == 0 means it's a directional light
        # 保存或更新 `lights` 的值。
        self.lights = [np.array([np.cos(t), np.sin(t), 0.0, 0.0])
            for t in 0.2 + np.linspace(0, 2*np.pi, 4)[:-1]]

    # call only once GL context is ready
    # 定义函数 `initialize`。
    def initialize(self):
        # 调用 `glShadeModel` 执行当前处理。
        glShadeModel(GL_SMOOTH)
        # 调用 `glEnable` 执行当前处理。
        glEnable(GL_LIGHTING)

        #glFogi(GL_FOG_MODE, GL_LINEAR)
        #glFogf(GL_FOG_START, 20.0) # Fog Start Depth
        #glFogf(GL_FOG_END, 100.0) # Fog End Depth
        #glEnable(GL_FOG)

        # 同时更新 `amb`, `diff`, `spec` 等变量。
        amb, diff, spec = (1.0 / len(self.lights)) * np.array([0.4, 1.2, 0.5])
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i, light in enumerate(self.lights):
            # TODO fix lights in world space instead of camera space
            # 调用 `glLightfv` 执行当前处理。
            glLightfv(GL_LIGHT0 + i, GL_POSITION, (GLfloat * 4)(*light))
            # 调用 `glLightfv` 执行当前处理。
            glLightfv(GL_LIGHT0 + i, GL_AMBIENT, (GLfloat * 4)(amb, amb, amb, 1))
            # 调用 `glLightfv` 执行当前处理。
            glLightfv(GL_LIGHT0 + i, GL_DIFFUSE, (GLfloat * 4)(diff, diff, diff, 1))
            # 调用 `glLightfv` 执行当前处理。
            glLightfv(GL_LIGHT0 + i, GL_SPECULAR, (GLfloat * 4)(spec, spec, spec, 1))
            # 调用 `glEnable` 执行当前处理。
            glEnable(GL_LIGHT0 + i)


# 定义函数 `draw`。
def draw(scene, camera, target):

    # 调用 `bind` 执行当前处理。
    target.bind() # sets viewport

    # 同时更新 `r`, `g`, `b` 等变量。
    r, g, b = scene.bgcolor
    # 调用 `glClearColor` 执行当前处理。
    glClearColor(r, g, b, 1.0)
    # 调用 `glFrontFace` 执行当前处理。
    glFrontFace(GL_CCW)
    # 调用 `glCullFace` 执行当前处理。
    glCullFace(GL_BACK)
    # 调用 `glEnable` 执行当前处理。
    glEnable(GL_CULL_FACE)
    # 调用 `glEnable` 执行当前处理。
    glEnable(GL_DEPTH_TEST)
    # 调用 `glEnable` 执行当前处理。
    glEnable(GL_NORMALIZE)
    # 调用 `glClear` 执行当前处理。
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 调用 `_matrix` 执行当前处理。
    camera._matrix(target.shape)
    #view = (GLfloat * 16)()
    #glGetFloatv(GL_MODELVIEW_MATRIX, view)
    #view = np.array(view).reshape((4,4)).T

    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for batch in scene.batches:
        # 调用 `draw` 执行当前处理。
        batch.draw()

    # 调用 `finish` 执行当前处理。
    target.finish()


# 定义类 `SceneNode`。
class SceneNode(object):
    # 定义函数 `_build_children`。
    def _build_children(self, batch):
        # hack - should go somewhere else
        # 根据条件决定是否进入当前分支。
        if not isinstance(self.children, type([])):
            # 保存或更新 `children` 的值。
            self.children = [self.children]
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for c in self.children:
            # 调用 `build` 执行当前处理。
            c.build(batch, self.pyg_grp)

    # default impl
    # 定义函数 `collide_sphere`。
    def collide_sphere(self, x, radius):
        # 返回当前函数的结果。
        return any(c.collide_sphere(x, radius) for c in self.children)

# 定义类 `World`。
class World(SceneNode):
    # 定义函数 `__init__`。
    def __init__(self, children):
        # 保存或更新 `children` 的值。
        self.children = children
        # 保存或更新 `pyg_grp` 的值。
        self.pyg_grp = None

    # 定义函数 `build`。
    def build(self, batch):
        #batch.add(3, GL_LINES, None, ('v2f', (0, 0, 5, 0, 0, 5)))
        # 调用 `_build_children` 执行当前处理。
        self._build_children(batch)

# 定义类 `Transform`。
class Transform(SceneNode):
    # 定义函数 `__init__`。
    def __init__(self, transform, children):
        # 保存或更新 `t` 的值。
        self.t = transform
        # 保存或更新 `mat_inv` 的值。
        self.mat_inv = np.linalg.inv(transform)
        # 保存或更新 `children` 的值。
        self.children = children

    # 定义函数 `build`。
    def build(self, batch, parent):
        # 保存或更新 `pyg_grp` 的值。
        self.pyg_grp = _PygTransform(self.t, parent=parent)
        # 调用 `_build_children` 执行当前处理。
        self._build_children(batch)
        # 返回当前函数的结果。
        return self.pyg_grp

    # 定义函数 `set_transform`。
    def set_transform(self, t):
        # 调用 `set_matrix` 执行当前处理。
        self.pyg_grp.set_matrix(t)
        # 保存或更新 `mat_inv` 的值。
        self.mat_inv = np.linalg.inv(t)

    # 定义函数 `set_transform_nocollide`。
    def set_transform_nocollide(self, t):
        # 调用 `set_matrix` 执行当前处理。
        self.pyg_grp.set_matrix(t)

    # 定义函数 `set_transform_and_color`。
    def set_transform_and_color(self, t, rgba):
        # 调用 `set_matrix` 执行当前处理。
        self.pyg_grp.set_matrix(t)
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for child in self.children:
            # 调用 `set_rgba` 执行当前处理。
            child.set_rgba(rgba[0], rgba[1], rgba[2], rgba[3])

    # 定义函数 `collide_sphere`。
    def collide_sphere(self, x, radius):
        # 保存或更新 `xh` 的值。
        xh = [x[0], x[1], x[2], 1]
        # 保存或更新 `xlocal` 的值。
        xlocal = np.matmul(self.mat_inv, xh)[:3]
        # 保存或更新 `rlocal` 的值。
        rlocal = radius * self.mat_inv[0,0]
        # 返回当前函数的结果。
        return any(c.collide_sphere(xlocal, rlocal) for c in self.children)

# 定义类 `BackToFront`。
class BackToFront(SceneNode):
    # 定义函数 `__init__`。
    def __init__(self, children):
        # 保存或更新 `children` 的值。
        self.children = children

    # 定义函数 `build`。
    def build(self, batch, parent):
        # 保存或更新 `pyg_grp` 的值。
        self.pyg_grp = pyglet.graphics.Group(parent=parent)
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i, c in enumerate(self.children):
            # 保存或更新 `ordering` 的值。
            ordering = pyglet.graphics.OrderedGroup(i, parent=self.pyg_grp)
            # 调用 `build` 执行当前处理。
            c.build(batch, ordering)
        # 返回当前函数的结果。
        return self.pyg_grp

# 定义类 `Color`。
class Color(SceneNode):
    # 定义函数 `__init__`。
    def __init__(self, color, children):
        # 保存或更新 `color` 的值。
        self.color = color
        # 保存或更新 `children` 的值。
        self.children = children

    # 定义函数 `build`。
    def build(self, batch, parent):
        # 保存或更新 `pyg_grp` 的值。
        self.pyg_grp = _PygColor(self.color, parent=parent)
        # 调用 `_build_children` 执行当前处理。
        self._build_children(batch)
        # 返回当前函数的结果。
        return self.pyg_grp

    # 定义函数 `set_rgb`。
    def set_rgb(self, r, g, b):
        # 调用 `set_rgb` 执行当前处理。
        self.pyg_grp.set_rgb(r, g, b)

    # 定义函数 `set_rgba`。
    def set_rgba(self, r, g, b, a):
        # 调用 `set_rgba` 执行当前处理。
        self.pyg_grp.set_rgba(r, g, b, a)

# 定义函数 `transform_and_color`。
def transform_and_color(transform, color, children):
    # 返回当前函数的结果。
    return Transform(transform, Color(color, children))

# 定义函数 `transform_and_dual_color`。
def transform_and_dual_color(transform, color_1, color_2, children_1, children_2):
    # 返回当前函数的结果。
    return Transform(transform, [Color(color_1, children_1), Color(color_2, children_2)])

# 保存或更新 `TEX_CHECKER` 的值。
TEX_CHECKER = 0
# 保存或更新 `TEX_XOR` 的值。
TEX_XOR = 1
# 保存或更新 `TEX_NOISE_GAUSSIAN` 的值。
TEX_NOISE_GAUSSIAN = 2
# 保存或更新 `TEX_NOISE_PERLIN` 的值。
TEX_NOISE_PERLIN = 3
# 保存或更新 `TEX_OILY` 的值。
TEX_OILY = 4
# 保存或更新 `TEX_VORONOI` 的值。
TEX_VORONOI = 5

# 定义函数 `random_textype`。
def random_textype():
    # 返回当前函数的结果。
    return np.random.randint(TEX_VORONOI + 1)

# 定义类 `ProceduralTexture`。
class ProceduralTexture(SceneNode):
    # 定义函数 `__init__`。
    def __init__(self, style, scale, children):
        # 保存或更新 `children` 的值。
        self.children = children
        # linear is default, those w/ nearest must overwrite
        # 保存或更新 `mag_filter` 的值。
        self.mag_filter = GL_LINEAR
        # 根据条件决定是否进入当前分支。
        if style == TEX_CHECKER:
            # 保存或更新 `image` 的值。
            image = np.zeros((256, 256))
            # 保存或更新 `image[:128,:128]` 的值。
            image[:128,:128] = 1.0
            # 保存或更新 `image[128:,128:]` 的值。
            image[128:,128:] = 1.0
            # 保存或更新 `mag_filter` 的值。
            self.mag_filter = GL_NEAREST
        # 当上一分支不满足时，继续判断新的条件。
        elif style == TEX_XOR:
            # 同时更新 `x`, `y` 等变量。
            x, y = np.meshgrid(range(256), range(256))
            # 保存或更新 `image` 的值。
            image = np.float32(np.bitwise_xor(np.uint8(x), np.uint8(y)))
            # 保存或更新 `mag_filter` 的值。
            self.mag_filter = GL_NEAREST
        # 当上一分支不满足时，继续判断新的条件。
        elif style == TEX_NOISE_GAUSSIAN:
            # 保存或更新 `nz` 的值。
            nz = np.random.normal(size=(256,256))
            # 保存或更新 `image` 的值。
            image = np.clip(nz, -3, 3)
        # 当上一分支不满足时，继续判断新的条件。
        elif style == TEX_NOISE_PERLIN:
            # 保存或更新 `t` 的值。
            t = np.linspace(0, 1, 256)
            # 保存或更新 `nzfun` 的值。
            nzfun = lambda x, y: noise.pnoise2(x, y,
                octaves=10, persistence=0.8, repeatx=1, repeaty=1)
            # 保存或更新 `image` 的值。
            image = np.vectorize(nzfun)(*np.meshgrid(t, t))
        # 当上一分支不满足时，继续判断新的条件。
        elif style == TEX_OILY:
            # from upvector.com "Intro to Procedural Textures"
            # 保存或更新 `t` 的值。
            t = np.linspace(0, 4, 256)
            # 保存或更新 `nzfun` 的值。
            nzfun = lambda x, y: noise.snoise2(x, y,
                octaves=10, persistence=0.45, repeatx=4, repeaty=4)
            # 保存或更新 `nz` 的值。
            nz = np.vectorize(nzfun)(*np.meshgrid(t, t))

            # 保存或更新 `t` 的值。
            t = np.linspace(0, 20*np.pi, 257)[:-1]
            # 同时更新 `x`, `y` 等变量。
            x, y = np.meshgrid(t, t)
            # 保存或更新 `image` 的值。
            image = np.sin(x + 8*nz)
        # 当上一分支不满足时，继续判断新的条件。
        elif style == TEX_VORONOI:
            # 保存或更新 `npts` 的值。
            npts = 64
            # 保存或更新 `points` 的值。
            points = np.random.uniform(size=(npts, 2))
            # make it tile
            # 保存或更新 `shifts` 的值。
            shifts = itertools.product([-1, 0, 1], [-1, 0, 1])
            # 保存或更新 `points` 的值。
            points = np.vstack([points + shift for shift in shifts])
            # 保存或更新 `unlikely` 的值。
            unlikely = np.any(np.logical_or(points < -0.25, points > 1.25), axis=1)
            # 保存或更新 `points` 的值。
            points = np.delete(points, np.where(unlikely), axis=0)
            # 保存或更新 `a` 的值。
            a = np.full((256, 256), np.inf)
            # 保存或更新 `t` 的值。
            t = np.linspace(0, 1, 256)
            # 同时更新 `x`, `y` 等变量。
            x, y = np.meshgrid(t, t)
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for p in points:
                # 保存或更新 `dist2` 的值。
                dist2 = (x - p[0])**2 + (y - p[1])**2
                # 保存或更新 `a` 的值。
                a = np.minimum(a, dist2)
            # 保存或更新 `image` 的值。
            image = np.sqrt(a)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 主动抛出异常以中止或提示错误。
            raise KeyError("style does not exist")

        # 同时更新 `low`, `high` 等变量。
        low, high = 255.0 * scale[0], 255.0 * scale[1]
        # 调用 `_scale_to_inplace` 执行当前处理。
        _scale_to_inplace(image, low, high)
        # 保存或更新 `tex` 的值。
        self.tex = _np2tex(image)

    # 定义函数 `build`。
    def build(self, batch, parent):
        # 保存或更新 `pyg_grp` 的值。
        self.pyg_grp = _PygTexture(tex=self.tex,
            mag_filter=self.mag_filter, parent=parent)
        # 调用 `_build_children` 执行当前处理。
        self._build_children(batch)
        # 返回当前函数的结果。
        return self.pyg_grp


#
# these functions return 4x4 rotation matrix suitable to construct Transform
# or to mutate Transform via set_matrix
#
# 定义函数 `scale`。
def scale(s):
    # 返回当前函数的结果。
    return np.diag([s, s, s, 1.0])

# 定义函数 `translate`。
def translate(x):
    # 保存或更新 `r` 的值。
    r = np.eye(4)
    # 保存或更新 `r[:3,3]` 的值。
    r[:3,3] = x
    # 返回当前函数的结果。
    return r

# 定义函数 `trans_and_rot`。
def trans_and_rot(t, r):
    # 保存或更新 `m` 的值。
    m = np.eye(4)
    # 保存或更新 `m[:3,:3]` 的值。
    m[:3,:3] = r
    # 保存或更新 `m[:3,3]` 的值。
    m[:3,3] = t
    # 返回当前函数的结果。
    return m

# 定义函数 `rotz`。
def rotz(theta):
    # 保存或更新 `r` 的值。
    r = np.eye(4)
    # 保存或更新 `r[:2,:2]` 的值。
    r[:2,:2] = _rot2d(theta)
    # 返回当前函数的结果。
    return r

# 定义函数 `roty`。
def roty(theta):
    # 保存或更新 `r` 的值。
    r = np.eye(4)
    # 保存或更新 `r2d` 的值。
    r2d = _rot2d(theta)
    # 保存或更新 `r[[0,0,2,2],[0,2,0,2]]` 的值。
    r[[0,0,2,2],[0,2,0,2]] = _rot2d(theta).flatten()
    # 返回当前函数的结果。
    return r

# 定义函数 `rotx`。
def rotx(theta):
    # 保存或更新 `r` 的值。
    r = np.eye(4)
    # 保存或更新 `r[1:3,1:3]` 的值。
    r[1:3,1:3] = _rot2d(theta)
    # 返回当前函数的结果。
    return r

# 定义类 `_PygTransform`。
class _PygTransform(pyglet.graphics.Group):
    # 定义函数 `__init__`。
    def __init__(self, transform=np.eye(4), parent=None):
        # 调用 `super` 执行当前处理。
        super().__init__(parent)
        # 调用 `set_matrix` 执行当前处理。
        self.set_matrix(transform)

    # 定义函数 `set_matrix`。
    def set_matrix(self, transform):
        # 断言当前条件成立，用于保护运行假设。
        assert transform.shape == (4, 4)
        # 断言当前条件成立，用于保护运行假设。
        assert np.all(transform[3,:] == [0, 0, 0, 1])
        # 保存或更新 `matrix_raw` 的值。
        self.matrix_raw = (GLfloat * 16)(*transform.T.flatten())

    # 定义函数 `set_state`。
    def set_state(self):
        # 调用 `glPushMatrix` 执行当前处理。
        glPushMatrix()
        # 调用 `glMultMatrixf` 执行当前处理。
        glMultMatrixf(self.matrix_raw)

    # 定义函数 `unset_state`。
    def unset_state(self):
        # 调用 `glPopMatrix` 执行当前处理。
        glPopMatrix()

# 定义类 `_PygColor`。
class _PygColor(pyglet.graphics.Group):
    # 定义函数 `__init__`。
    def __init__(self, color, parent=None):
        # 调用 `super` 执行当前处理。
        super().__init__(parent)
        # 根据条件决定是否进入当前分支。
        if len(color) == 3:
            # 调用 `set_rgb` 执行当前处理。
            self.set_rgb(*color)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 调用 `set_rgba` 执行当前处理。
            self.set_rgba(*color)

    # 定义函数 `set_rgb`。
    def set_rgb(self, r, g, b):
        # 调用 `set_rgba` 执行当前处理。
        self.set_rgba(r, g, b, 1.0)

    # 定义函数 `set_rgba`。
    def set_rgba(self, r, g, b, a):
        # 保存或更新 `dcolor` 的值。
        self.dcolor = (GLfloat * 4)(r, g, b, a)
        # 保存或更新 `spec_whiteness` 的值。
        spec_whiteness = 0.8
        # 同时更新 `r`, `g`, `b` 等变量。
        r, g, b = (1.0 - spec_whiteness) * np.array([r, g, b]) + spec_whiteness
        # 保存或更新 `scolor` 的值。
        self.scolor = (GLfloat * 4)(r, g, b, a)

    # 定义函数 `set_state`。
    def set_state(self):
        # 根据条件决定是否进入当前分支。
        if self.dcolor[-1] < 1.0:
            # 调用 `glEnable` 执行当前处理。
            glEnable(GL_BLEND)
            # 调用 `glBlendFunc` 执行当前处理。
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # 调用 `glMaterialfv` 执行当前处理。
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, self.dcolor)
        # 调用 `glMaterialfv` 执行当前处理。
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, self.scolor)
        # 调用 `glMaterialfv` 执行当前处理。
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, (GLfloat)(8.0))

    # 定义函数 `unset_state`。
    def unset_state(self):
        # 根据条件决定是否进入当前分支。
        if self.dcolor[-1] < 1.0:
            # 调用 `glDisable` 执行当前处理。
            glDisable(GL_BLEND)

# 定义类 `_PygTexture`。
class _PygTexture(pyglet.graphics.Group):
    # 定义函数 `__init__`。
    def __init__(self, tex, mag_filter, parent=None):
        # 保存或更新 `super().__init__(parent` 的值。
        super().__init__(parent=parent)

        # 保存或更新 `tex` 的值。
        self.tex = tex
        # 调用 `glBindTexture` 执行当前处理。
        glBindTexture(GL_TEXTURE_2D, self.tex.id)
        # 调用 `glTexParameteri` 执行当前处理。
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        # 调用 `glTexParameteri` 执行当前处理。
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        # 调用 `glGenerateMipmap` 执行当前处理。
        glGenerateMipmap(GL_TEXTURE_2D);
        # 调用 `glTexParameteri` 执行当前处理。
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter)
        # 调用 `glTexParameteri` 执行当前处理。
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

        # anisotropic texturing helps a lot with checkerboard floors
        # 保存或更新 `anisotropy` 的值。
        anisotropy = (GLfloat)()
        # 调用 `glGetFloatv` 执行当前处理。
        glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy)
        # 调用 `glTexParameterf` 执行当前处理。
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy)

    # 定义函数 `set_state`。
    def set_state(self):
        # 调用 `glMaterialfv` 执行当前处理。
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (GLfloat * 4)(1,1,1,1))
        # 调用 `glMaterialfv` 执行当前处理。
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (GLfloat * 4)(1,1,1,1))
        # 调用 `glEnable` 执行当前处理。
        glEnable(self.tex.target)
        # 调用 `glBindTexture` 执行当前处理。
        glBindTexture(self.tex.target, self.tex.id)

    # 定义函数 `unset_state`。
    def unset_state(self):
        # 调用 `glDisable` 执行当前处理。
        glDisable(self.tex.target)


# 定义类 `_PygAlphaBlending`。
class _PygAlphaBlending(pyglet.graphics.Group):
    # 定义函数 `__init__`。
    def __init__(self, parent=None):
        # 保存或更新 `super().__init__(parent` 的值。
        super().__init__(parent=parent)

    # 定义函数 `set_state`。
    def set_state(self):
        # 调用 `glEnable` 执行当前处理。
        glEnable(GL_BLEND)
        # 调用 `glBlendFunc` 执行当前处理。
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # 定义函数 `unset_state`。
    def unset_state(self):
        # 调用 `glDisable` 执行当前处理。
        glDisable(GL_BLEND)

# 保存或更新 `Batch` 的值。
Batch = pyglet.graphics.Batch

# we only implement collision detection between primitives and spheres.
# the world-coordinate sphere is transformed according to the scene graph
# into the primitive's canonical coordinate system.
# this simplifies the math a lot, although it might be less efficient
# than directly testing the sphere against the transformed primitive
# in world coordinates.
# 定义类 `SphereCollision`。
class SphereCollision(object):
    # 定义函数 `__init__`。
    def __init__(self, radius):
        # 保存或更新 `radius` 的值。
        self.radius = radius
    # 定义函数 `collide_sphere`。
    def collide_sphere(self, x, radius):
        # 保存或更新 `c` 的值。
        c = np.sum(x ** 2) < (self.radius + radius) ** 2
        # 根据条件决定是否进入当前分支。
        if c: print("collided with sphere")
        # 返回当前函数的结果。
        return c

# 定义类 `AxisBoxCollision`。
class AxisBoxCollision(object):
    # 定义函数 `__init__`。
    def __init__(self, corner0, corner1):
        # 同时更新 `corner0`, `corner1` 等变量。
        self.corner0, self.corner1 = corner0, corner1
    # 定义函数 `collide_sphere`。
    def collide_sphere(self, x, radius):
        # 保存或更新 `nearest` 的值。
        nearest = np.maximum(self.corner0, np.minimum(x, self.corner1))
        # 保存或更新 `c` 的值。
        c = np.sum((x - nearest)**2) < radius**2
        # 根据条件决定是否进入当前分支。
        if c: print("collided with box")
        # 返回当前函数的结果。
        return c

# 定义类 `CapsuleCollision`。
class CapsuleCollision(object):
    # 定义函数 `__init__`。
    def __init__(self, radius, height):
        # 同时更新 `radius`, `height` 等变量。
        self.radius, self.height = radius, height
    # 定义函数 `collide_sphere`。
    def collide_sphere(self, x, radius):
        # 保存或更新 `z` 的值。
        z = min(max(0, x[2]), self.height)
        # 保存或更新 `nearest` 的值。
        nearest = [0, 0, z]
        # 保存或更新 `c` 的值。
        c = np.sum((x - nearest)**2) < (self.radius + radius)**2
        # 根据条件决定是否进入当前分支。
        if c: print("collided with capsule")
        # 返回当前函数的结果。
        return c

#
# these are the 3d primitives that can be added to a pyglet.graphics.Batch.
# construct them with the shape functions below.
#
# 定义类 `BatchElement`。
class BatchElement(SceneNode):
    # 定义函数 `build`。
    def build(self, batch, parent):
        # 保存或更新 `batch_args[2]` 的值。
        self.batch_args[2] = parent
        # 调用 `add` 执行当前处理。
        batch.add(*self.batch_args)

    # 定义函数 `collide_sphere`。
    def collide_sphere(self, x, radius):
        # 根据条件决定是否进入当前分支。
        if self.collider is not None:
            # 返回当前函数的结果。
            return self.collider.collide_sphere(x, radius)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 返回当前函数的结果。
            return False

# 定义类 `Mesh`。
class Mesh(BatchElement):
    # 定义函数 `__init__`。
    def __init__(self, verts, normals=None, st=None, collider=None):
        # 根据条件决定是否进入当前分支。
        if len(verts.shape) != 2 or verts.shape[1] != 3:
            # 主动抛出异常以中止或提示错误。
            raise ValueError('verts must be an N x 3 NumPy array')

        # 保存或更新 `N` 的值。
        N = verts.shape[0]
        # 断言当前条件成立，用于保护运行假设。
        assert int(N) % 3 == 0

        # 根据条件决定是否进入当前分支。
        if st is not None:
            # 断言当前条件成立，用于保护运行假设。
            assert st.shape == (N, 2)

        # 根据条件决定是否进入当前分支。
        if normals is None:
            # compute normals implied by triangle faces
            # 保存或更新 `normals` 的值。
            normals = deepcopy(verts)

            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i in range(0, N, 3):
                # 同时更新 `v0`, `v1`, `v2` 等变量。
                v0, v1, v2 = verts[i:(i+3),:]
                # 同时更新 `d0`, `d1` 等变量。
                d0, d1 = (v1 - v0), (v2 - v1)
                # 保存或更新 `n` 的值。
                n = _normalize(np.cross(d0, d1))
                # 保存或更新 `normals[i:(i+3),:]` 的值。
                normals[i:(i+3),:] = n

        # 保存或更新 `batch_args` 的值。
        self.batch_args = [N, pyglet.gl.GL_TRIANGLES, None,
            ('v3f/static', list(verts.flatten())),
            ('n3f/static', list(normals.flatten())),
        ]
        # 根据条件决定是否进入当前分支。
        if st is not None:
            # 调用 `append` 执行当前处理。
            self.batch_args.append(('t2f/static', list(st.flatten())))
        # 保存或更新 `collider` 的值。
        self.collider = collider

# 定义类 `TriStrip`。
class TriStrip(BatchElement):
    # 定义函数 `__init__`。
    def __init__(self, verts, normals, collider=None):
        # 同时更新 `N`, `dim` 等变量。
        N, dim = verts.shape
        # 断言当前条件成立，用于保护运行假设。
        assert dim == 3
        # 断言当前条件成立，用于保护运行假设。
        assert normals.shape == verts.shape

        # 保存或更新 `batch_args` 的值。
        self.batch_args = [N, pyglet.gl.GL_TRIANGLE_STRIP, None,
            ('v3f/static', list(verts.flatten())),
            ('n3f/static', list(normals.flatten()))
        ]
        # 保存或更新 `collider` 的值。
        self.collider = collider

# 定义类 `TriFan`。
class TriFan(BatchElement):
    # 定义函数 `__init__`。
    def __init__(self, verts, normals, collider=None):
        # 同时更新 `N`, `dim` 等变量。
        N, dim = verts.shape
        # 断言当前条件成立，用于保护运行假设。
        assert dim == 3
        # 断言当前条件成立，用于保护运行假设。
        assert normals.shape == verts.shape

        # 保存或更新 `batch_args` 的值。
        self.batch_args = [N, pyglet.gl.GL_TRIANGLE_FAN, None,
            ('v3f/static', list(verts.flatten())),
            ('n3f/static', list(normals.flatten()))
        ]
        # 保存或更新 `collider` 的值。
        self.collider = collider

# a box centered on the origin
# 定义函数 `box`。
def box(x, y, z):
    # 保存或更新 `corner1` 的值。
    corner1 = np.array([x,y,z]) / 2
    # 保存或更新 `corner0` 的值。
    corner0 = -corner1
    # 保存或更新 `v` 的值。
    v = box_mesh(x, y, z)
    # 保存或更新 `collider` 的值。
    collider = AxisBoxCollision(corner0, corner1)
    # 返回当前函数的结果。
    return Mesh(v, collider=collider)

# Box representing the environment. Shortcut way to add walls during visualization
# 定义函数 `envBox`。
def envBox(x, y, z):
    # corner1 = np.array([-x, -y, z])
    # corner2 = np.array([x, y, 0])
    # 保存或更新 `corner1` 的值。
    corner1 = np.array([x/2, y/2, z])
    # 保存或更新 `corner2` 的值。
    corner2 = -corner1
    # 保存或更新 `corner2[2]` 的值。
    corner2[2] = 0
    # 保存或更新 `v` 的值。
    v = room_mesh(x, y, z)
    # 保存或更新 `collider` 的值。
    collider = AxisBoxCollision(corner1, corner2)
    # 返回当前函数的结果。
    return Mesh(v, collider=collider)

# cylinder sitting on xy plane pointing +z
# 定义函数 `cylinder`。
def cylinder(radius, height, sections):
    # 同时更新 `v`, `n` 等变量。
    v, n = cylinder_strip(radius, height, sections)
    # 保存或更新 `collider` 的值。
    collider = CapsuleCollision(radius, height)
    # 返回当前函数的结果。
    return TriStrip(v, n, collider=collider)

# cylinder sitting on xy plane pointing +x
# 定义函数 `rod`。
def rod(radius, length, sections):
    # 同时更新 `v`, `n` 等变量。
    v, n = cylinder_strip(radius, length, sections)
    # 保存或更新 `collider` 的值。
    collider = CapsuleCollision(radius, length)
    # 返回当前函数的结果。
    return TriStrip(v, n, collider=collider)

# cone sitting on xy plane pointing +z
# 定义函数 `cone`。
def cone(radius, height, sections):
    # TODO collision detectoin
    # 同时更新 `v`, `n` 等变量。
    v, n = cone_strip(radius, height, sections)
    # 返回当前函数的结果。
    return TriStrip(v, n)

# arrow sitting on xy plane pointing +z
# 定义函数 `arrow`。
def arrow(radius, height, sections):
    # 同时更新 `v`, `n` 等变量。
    v, n = arrow_strip(radius, height, sections)
    # 返回当前函数的结果。
    return TriStrip(v, n)

# sphere centered on origin, n tris will be about TODO * facets
# 定义函数 `sphere`。
def sphere(radius, facets, facet_range=None):
    # 同时更新 `v`, `n` 等变量。
    v, n = sphere_strip(radius, facets, facet_range)
    # 保存或更新 `collider` 的值。
    collider = SphereCollision(radius)
    # 返回当前函数的结果。
    return TriStrip(v, n, collider=collider)

# square in xy plane centered on origin
# dim: (w, h)
# srange, trange: desired min/max (s, t) tex coords
# 定义函数 `rect`。
def rect(dim, srange=(0,1), trange=(0,1)):
    # 保存或更新 `v` 的值。
    v = np.array([
        [1, 1, 0], [-1, 1, 0], [1, -1, 0],
        [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])
    # 保存或更新 `v` 的值。
    v = np.matmul(v, np.diag([dim[0] / 2.0, dim[1] / 2.0, 0]))
    # 保存或更新 `n` 的值。
    n = _withz(0 * v, 1)
    # 同时更新 `s0`, `s1` 等变量。
    s0, s1 = srange
    # 同时更新 `t0`, `t1` 等变量。
    t0, t1 = trange
    # 保存或更新 `st` 的值。
    st = np.array([
        [s1, t1], [s0, t1], [s1, t0],
        [s0, t1], [s0, t0], [s1, t0]])
    # 返回当前函数的结果。
    return Mesh(v, n, st)


# 定义函数 `circle`。
def circle(radius, facets):
    # 同时更新 `v`, `n` 等变量。
    v, n = circle_fan(radius, facets)
    # 返回当前函数的结果。
    return TriFan(v, n)

#
# low-level primitive builders. return vertex/normal/texcoord arrays.
# good if you want to apply transforms directly to the points, etc.
#

# box centered on origin with given dimensions.
# no normals, but Mesh ctor will estimate them perfectly
# 定义函数 `box_mesh`。
def box_mesh(x, y, z):
    # 保存或更新 `vtop` 的值。
    vtop = np.array([[x, y, z], [x, -y, z], [-x, -y, z], [-x, y, z]])
    # 保存或更新 `vbottom` 的值。
    vbottom = deepcopy(vtop)
    # 保存或更新 `vbottom[:,2]` 的值。
    vbottom[:,2] = -vbottom[:,2]
    # 保存或更新 `v` 的值。
    v = 0.5 * np.concatenate([vtop, vbottom], axis=0)
    # 保存或更新 `t` 的值。
    t = np.array([[1, 3, 2,], [1, 4, 3,], [1, 2, 5,], [2, 6, 5,], [2, 3, 6,], [3, 7, 6,], [3, 4, 8,], [3, 8, 7,], [4, 1, 8,], [1, 5, 8,], [5, 6, 7,], [5, 7, 8,]]) - 1
    # 保存或更新 `t` 的值。
    t = t.flatten()
    # 保存或更新 `v` 的值。
    v = v[t,:]
    # 返回当前函数的结果。
    return v

# need a different mesh function for the envBox
# 定义函数 `room_mesh`。
def room_mesh(x, y, z):
    # 保存或更新 `vtop` 的值。
    vtop = np.array([[x/2, y/2, z], [x/2, -y/2, z], [-x/2, -y/2, z], [-x/2, y/2, z]])
    # 保存或更新 `vbottom` 的值。
    vbottom = deepcopy(vtop)
    # 保存或更新 `vbottom[:, 2]` 的值。
    vbottom[:, 2] = -1.0
    # 保存或更新 `v` 的值。
    v = np.concatenate([vtop, vbottom], axis=0)
    # triangle meshes
    # 保存或更新 `t` 的值。
    t = np.array(
        [[1, 2, 3, ], [1, 3, 4, ], [1, 5, 2, ], [2, 5, 6, ], [2, 6, 3, ], [3, 6, 7, ], [3, 8, 4, ], [3, 7, 8, ],
         [4, 8, 1, ], [1, 8, 5, ], [5, 7, 6, ], [5, 8, 7, ]]) - 1
    # 保存或更新 `t` 的值。
    t = t.flatten()
    # 保存或更新 `v` 的值。
    v = v[t, :]
    # 返回当前函数的结果。
    return v

# circle in the x-y plane
# 定义函数 `circle_fan`。
def circle_fan(radius, sections):
    # 保存或更新 `t` 的值。
    t = np.linspace(0, 2 * np.pi, sections + 1)[:,None]
    # 保存或更新 `x` 的值。
    x = radius * np.cos(t)
    # 保存或更新 `y` 的值。
    y = radius * np.sin(t)
    # 保存或更新 `v` 的值。
    v = np.hstack([x, y, 0*t])
    # 保存或更新 `v` 的值。
    v = np.vstack([[0, 0, 0], v])
    # 保存或更新 `n` 的值。
    n = _withz(0 * v, 1)
    # 返回当前函数的结果。
    return v, n

# cylinder sitting on the x-y plane
# 定义函数 `cylinder_strip`。
def cylinder_strip(radius, height, sections):
    # 保存或更新 `t` 的值。
    t = np.linspace(0, 2 * np.pi, sections + 1)[:,None]
    # 保存或更新 `x` 的值。
    x = radius * np.cos(t)
    # 保存或更新 `y` 的值。
    y = radius * np.sin(t)

    # 保存或更新 `base` 的值。
    base = np.hstack([x, y, 0*t])
    # 保存或更新 `top` 的值。
    top = np.hstack([x, y, height + 0*t])
    # 保存或更新 `strip_sides` 的值。
    strip_sides = _to_strip(np.hstack([base[:,None,:], top[:,None,:]]))
    # 保存或更新 `normals_sides` 的值。
    normals_sides = _withz(strip_sides / radius, 0)

    # 定义函数 `make_cap`。
    def make_cap(circle, normal_z):
        # 保存或更新 `height` 的值。
        height = circle[0,2]
        # 保存或更新 `center` 的值。
        center = _withz(0 * circle, height)
        # 根据条件决定是否进入当前分支。
        if normal_z > 0:
            # 保存或更新 `strip` 的值。
            strip = _to_strip(np.hstack([circle[:,None,:], center[:,None,:]]))
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `strip` 的值。
            strip = _to_strip(np.hstack([center[:,None,:], circle[:,None,:]]))
        # 保存或更新 `normals` 的值。
        normals = _withz(0 * strip, normal_z)
        # 返回当前函数的结果。
        return strip, normals

    # 同时更新 `vbase`, `nbase` 等变量。
    vbase, nbase = make_cap(base, -1)
    # 同时更新 `vtop`, `ntop` 等变量。
    vtop, ntop = make_cap(top, 1)
    # 返回当前函数的结果。
    return (
        np.vstack([strip_sides, vbase, vtop]),
        np.vstack([normals_sides, nbase, ntop]))

# cone sitting on the x-y plane
# 定义函数 `cone_strip`。
def cone_strip(radius, height, sections):
    # 保存或更新 `t` 的值。
    t = np.linspace(0, 2 * np.pi, sections + 1)[:,None]
    # 保存或更新 `x` 的值。
    x = radius * np.cos(t)
    # 保存或更新 `y` 的值。
    y = radius * np.sin(t)
    # 保存或更新 `base` 的值。
    base = np.hstack([x, y, 0*t])

    # 保存或更新 `top` 的值。
    top = _withz(0 * base, height)
    # 保存或更新 `vside` 的值。
    vside = _to_strip(np.hstack([base[:,None,:], top[:,None,:]]))
    # 保存或更新 `base_tangent` 的值。
    base_tangent = np.cross(_npa(0, 0, 1), base)
    # 保存或更新 `top_to_base` 的值。
    top_to_base = base - top
    # 保存或更新 `normals` 的值。
    normals = _normalize(np.cross(top_to_base, base_tangent))
    # 保存或更新 `nside` 的值。
    nside = _to_strip(np.hstack([normals[:,None,:], normals[:,None,:]]))

    # 保存或更新 `base_ctr` 的值。
    base_ctr = 0 * base
    # 保存或更新 `vbase` 的值。
    vbase = _to_strip(np.hstack([base_ctr[:,None,:], base[:,None,:]]))
    # 保存或更新 `nbase` 的值。
    nbase = _withz(0 * vbase, -1)

    # 返回当前函数的结果。
    return np.vstack([vside, vbase]), np.vstack([nside, nbase])

# sphere centered on origin
# 定义函数 `sphere_strip`。
def sphere_strip(radius, resolution, resolution_range=None):
    # 保存或更新 `t` 的值。
    t = np.linspace(-1, 1, resolution)
    # 同时更新 `u`, `v` 等变量。
    u, v = np.meshgrid(t, t)
    # 保存或更新 `vtx` 的值。
    vtx = []
    # 保存或更新 `panel` 的值。
    panel = np.zeros((resolution, resolution, 3))
    # 保存或更新 `inds` 的值。
    inds = list(range(3))
    # 保存或更新 `res_range` 的值。
    res_range = (0, resolution-1) if not resolution_range else resolution_range
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i in range(3):
        # 保存或更新 `panel[:,:,inds[0]]` 的值。
        panel[:,:,inds[0]] = u
        # 保存或更新 `panel[:,:,inds[1]]` 的值。
        panel[:,:,inds[1]] = v
        # 保存或更新 `panel[:,:,inds[2]]` 的值。
        panel[:,:,inds[2]] = 1
        # 保存或更新 `norms` 的值。
        norms = np.linalg.norm(panel, axis=2)
        # 保存或更新 `panel` 的值。
        panel = panel / norms[:,:,None]
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for _ in range(2):
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for j in range(res_range[0], res_range[1]):
                # 保存或更新 `strip` 的值。
                strip = deepcopy(panel[[j,j+1],:,:].transpose([1,0,2]).reshape((-1,3)))
                # 保存或更新 `degen0` 的值。
                degen0 = deepcopy(strip[0,:])
                # 保存或更新 `degen1` 的值。
                degen1 = deepcopy(strip[-1,:])
                # 调用 `extend` 执行当前处理。
                vtx.extend([degen0, strip, degen1])
            # 保存或更新 `panel` 的值。
            panel *= -1
            # 保存或更新 `panel` 的值。
            panel = np.flip(panel, axis=1)
        # 保存或更新 `inds` 的值。
        inds = [inds[-1]] + inds[:-1]

    # 保存或更新 `n` 的值。
    n = np.vstack(vtx)
    # 保存或更新 `v` 的值。
    v = radius * n
    # 返回当前函数的结果。
    return v, n

# arrow sitting on x-y plane
# 定义函数 `arrow_strip`。
def arrow_strip(radius, height, facets):
    # 保存或更新 `cyl_r` 的值。
    cyl_r = radius
    # 保存或更新 `cyl_h` 的值。
    cyl_h = 0.75 * height
    # 保存或更新 `cone_h` 的值。
    cone_h = height - cyl_h
    # 保存或更新 `cone_half_angle` 的值。
    cone_half_angle = np.radians(30)
    # 保存或更新 `cone_r` 的值。
    cone_r = 1.5 * cone_h * np.tan(cone_half_angle)
    # 同时更新 `vcyl`, `ncyl` 等变量。
    vcyl, ncyl = cylinder_strip(cyl_r, cyl_h, facets)
    # 同时更新 `vcone`, `ncone` 等变量。
    vcone, ncone = cone_strip(cone_r, cone_h, facets)
    # 保存或更新 `vcone[:,2]` 的值。
    vcone[:,2] += cyl_h
    # 保存或更新 `v` 的值。
    v = np.vstack([vcyl, vcone])
    # 保存或更新 `n` 的值。
    n = np.vstack([ncyl, ncone])
    # 返回当前函数的结果。
    return v, n


#
# private helper functions, not part of API
#
# 定义函数 `_npa`。
def _npa(*args):
    # 返回当前函数的结果。
    return np.array(args)

# 定义函数 `_normalize`。
def _normalize(x):
    # 根据条件决定是否进入当前分支。
    if len(x.shape) == 1:
        # 返回当前函数的结果。
        return x / np.linalg.norm(x)
    # 当上一分支不满足时，继续判断新的条件。
    elif len(x.shape) == 2:
        # 返回当前函数的结果。
        return x / np.linalg.norm(x, axis=1)[:,None]
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 断言当前条件成立，用于保护运行假设。
        assert False

# 定义函数 `_withz`。
def _withz(a, z):
    # 保存或更新 `b` 的值。
    b = 0 + a
    # 保存或更新 `b[:,2]` 的值。
    b[:,2] = z
    # 返回当前函数的结果。
    return b

# 定义函数 `_rot2d`。
def _rot2d(theta):
    # 保存或更新 `c` 的值。
    c = np.cos(theta)
    # 保存或更新 `s` 的值。
    s = np.sin(theta)
    # 返回当前函数的结果。
    return np.array([[c, -s], [s, c]])

# add degenerate tris, convert from N x 2 x 3 to 2N+2 x 3
# 定义函数 `_to_strip`。
def _to_strip(strip):
    # 保存或更新 `s0` 的值。
    s0 = strip[0,0,:]
    # 保存或更新 `s1` 的值。
    s1 = strip[-1,-1,:]
    # 返回当前函数的结果。
    return np.vstack([s0, np.reshape(strip, (-1, 3)), s1])

# 定义函数 `_scale_to_inplace`。
def _scale_to_inplace(a, min1, max1):
    # 保存或更新 `id0` 的值。
    id0 = id(a)
    # 保存或更新 `min0` 的值。
    min0 = np.min(a.flatten())
    # 保存或更新 `max0` 的值。
    max0 = np.max(a.flatten())
    # 保存或更新 `scl` 的值。
    scl = (max1 - min1) / (max0 - min0)
    # 保存或更新 `shift` 的值。
    shift = - (scl * min0) + min1
    # 保存或更新 `a` 的值。
    a *= scl
    # 保存或更新 `a` 的值。
    a += shift
    # 断言当前条件成立，用于保护运行假设。
    assert id(a) == id0

# 定义函数 `_np2tex`。
def _np2tex(a):
    # TODO color
    # 同时更新 `w`, `h` 等变量。
    w, h = a.shape
    # 保存或更新 `b` 的值。
    b = np.uint8(a).tobytes()
    # 断言当前条件成立，用于保护运行假设。
    assert len(b) == w * h
    # 保存或更新 `img` 的值。
    img = pyglet.image.ImageData(w, h, "L", b)
    # 返回当前函数的结果。
    return img.get_texture()


