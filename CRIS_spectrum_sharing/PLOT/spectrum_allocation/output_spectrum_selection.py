import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 创建一个3x41的方格
time_steps = 41
num_subchannel = 3
plt.rc('font',family='Times New Roman')
# 创建一个新的图形
fig, ax = plt.subplots()

PU_csv = np.loadtxt("../SD3/User_CreateData/Test_PU_Spectrum_MDP.csv", delimiter=",")
usage_ls = []
for i in range(time_steps):
    usage = PU_csv[i][:3]
    for j in range(num_subchannel):
        if usage[j] != 0:
            usage_ls.append((i,j))

# 绘制方格和边框
for i in range(time_steps):
    for j in range(num_subchannel):
        square = Rectangle((i, j), width=1, height=1, edgecolor='black', linewidth=1)
        if (i, j) in usage_ls:
            square.set_facecolor('lightgray')
            # square.set_hatch('////')  # 调整斜线密度的参数
        else:
            square.set_facecolor('white')
        ax.add_patch(square)

# 设置坐标轴范围和刻度
ax.set_xlim(0, time_steps)
ax.set_ylim(0, num_subchannel)
ax.set_xticks(np.arange(time_steps) + 0.5)
ax.set_yticks(np.arange(num_subchannel) + 0.5)
ax.set_xticklabels(np.arange(time_steps))
ax.set_yticklabels(np.arange(num_subchannel))
ax.tick_params(axis='both', length=0, labelcolor='black', labelsize=8)
fig.set_size_inches(10, 2)
plt.subplots_adjust(bottom=0.2)

ax.set_xlabel('Time Steps')
ax.set_ylabel('Subchannel Index')

# 显示图形

plt.savefig("PU_usage.png", dpi=300)

#--------------------------------------------#
# SU1 usage
SU_1_usage = [0,2,0,2,0,2,2,0,2,0,0,2,2,0,2,0,0,2,0,1,2,2,2,2,2,2,2,0,1,2,0,2,2,0,2,0,0,2,0,0,2]
usage_ls = []
for i in range(time_steps):
    usage_ls.append((i,SU_1_usage[i]))

for i in range(time_steps):
    for j in range(num_subchannel):
        square = Rectangle((i, j), width=1, height=1, edgecolor='black', linewidth=1)
        if (i, j) in usage_ls:
            square.set_facecolor('white')
            square.set_hatch('///')  # 调整斜线密度的参数
            square.set_edgecolor('red')  # 设置斜线的颜色为红色
        else:
            square.set_facecolor('white')
        ax.add_patch(square)

# 设置坐标轴范围和刻度
ax.set_xlim(0, time_steps)
ax.set_ylim(0, num_subchannel)
ax.set_xticks(np.arange(time_steps) + 0.5)
ax.set_yticks(np.arange(num_subchannel) + 0.5)
ax.set_xticklabels(np.arange(time_steps))
ax.set_yticklabels(np.arange(num_subchannel))
ax.tick_params(axis='both', length=0, labelcolor='black', labelsize=8)
fig.set_size_inches(10, 2)

# 显示图形
plt.rc('font',family='Times New Roman')
plt.savefig("SU_1_usage.png", dpi=300)

#--------------------------------------------#
# SU2 usage
SU_2_usage = [1,0,2,0,2,0,0,2,0,2,1,0,0,2,0,1,1,0,2,2,0,0,0,0,0,0,0,2,2,0,1,0,0,2,0,2,2,0,2,2,0]
usage_ls = []
for i in range(time_steps):
    usage_ls.append((i,SU_2_usage[i]))

for i in range(time_steps):
    for j in range(num_subchannel):
        square = Rectangle((i, j), width=1, height=1, edgecolor='black', linewidth=1)
        if (i, j) in usage_ls:
            square.set_facecolor('white')
            square.set_hatch('\\\\\\')  # 调整斜线密度的参数
            square.set_edgecolor('blue')  # 设置斜线的颜色为红色
        else:
            square.set_facecolor('white')
        ax.add_patch(square)

# 设置坐标轴范围和刻度
ax.set_xlim(0, time_steps)
ax.set_ylim(0, num_subchannel)
ax.set_xticks(np.arange(time_steps) + 0.5)
ax.set_yticks(np.arange(num_subchannel) + 0.5)
ax.set_xticklabels(np.arange(time_steps))
ax.set_yticklabels(np.arange(num_subchannel))
ax.tick_params(axis='both', length=0, labelcolor='black', labelsize=8)
fig.set_size_inches(10, 2)

# 显示图形
plt.rc('font',family='Times New Roman')
plt.savefig("SU_2_usage.png", dpi=300)



