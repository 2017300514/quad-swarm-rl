# 中文注释副本；原始文件：gym_art/quadrotor_multi/tests/plot_v_value.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这不是单元测试，而是一个离线调试脚本：作者先在 Sample Factory 的 actor-critic 前向里打断点，
# 手工扫一维状态轴上的若干点，把 critic 输出的 value 拷出来，再用 Plotly 快速画出 1D 曲线。
# 它的作用是帮助判断某个单独状态维度变化时，价值函数是否具有符合直觉的形状。

import numpy as np

"""
# How to use:
1. Go to Sample-Factory, actor_critic.py: ActorCriticSeparateWeights: forward function
2. Set a debug point at the first line of the forward function.
    x = self.forward_head(normalized_obs_dict)
3. In PyCharm, go to Console, copy and past code below
tmp_score=[]
idx = []
for i in range(-20, 21):
    id_i = i * 0.1
    normalized_obs_dict['obs'][0][2]=id_i
    x = self.forward_head(normalized_obs_dict)
    x, new_rnn_states = self.forward_core(x, rnn_states)
    result = self.forward_tail(x, values_only, sample_actions=True)
    tmp_score.append(result['values'].item())
    idx.append(id_i)
print(tmp_score)
print(idx)
4. Copy and paste the print info and replace v_value dict below.
"""

import plotly.express as px

# 这里的 `x`/`y` 不是训练时自动生成的数据集，而是一次人工 debugger 扫描后的静态快照。
# 默认示例沿观测向量中的单个维度做切片，用来观察 value 随该维度变化时的大致单峰/单谷结构。
x = np.array(
    [
        -2.0, -1.9000000000000001, -1.8, -1.7000000000000002, -1.6, -1.5, -1.4000000000000001, -1.3,
        -1.2000000000000002, -1.1, -1.0, -0.9, -0.8, -0.7000000000000001, -0.6000000000000001, -0.5, -0.4,
        -0.30000000000000004, -0.2, -0.1, 0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001,
        0.7000000000000001, 0.8, 0.9, 1.0, 1.1, 1.2000000000000002, 1.3, 1.4000000000000001, 1.5, 1.6,
        1.7000000000000002, 1.8, 1.9000000000000001, 2.0,
    ]
)
y = np.array(
    [
        -0.715114951133728, -0.6702059507369995, -0.6279435157775879, -0.5909671187400818, -0.5606633424758911,
        -0.5370818376541138, -0.5195604562759399, -0.5072858333587646, -0.49951931834220886, -0.4955696165561676,
        -0.49465319514274597, -0.4958183467388153, -0.4981296956539154, -0.5009654760360718, -0.5033068656921387,
        -0.5005838871002197, -0.47910305857658386, -0.40353140234947205, -0.16471873223781586, 0.18300850689411163,
        0.2810545265674591, 0.2919209897518158, 0.2890877425670624, 0.28149619698524475, 0.27089038491249084,
        0.2577819526195526, 0.2421511560678482, 0.2236170917749405, 0.201537624001503, 0.17511744797229767,
        0.14352239668369293, 0.1060105711221695, 0.062122244387865067, 0.011940140277147293, -0.0436464361846447,
        -0.10285423696041107, -0.1632765382528305, -0.2224482148885727, -0.2784207761287689, -0.3300497233867645,
        -0.37695547938346863,
    ]
)

# 峰值位置通常被拿来判断“当前 critic 在这条一维切片上最偏好的状态区域”。
xmax = x[np.argmax(y)]
ymax = y.max()
text = "max value={:.5f}, z={:.2f}".format(ymax, xmax)

fig = px.scatter(x=x, y=y, title=text)
fig.show()
