import numpy as np
import matplotlib.pyplot as plt

data_X = ['11', '12', '13', '14', '15', '16']
data_Y = [130, 132, 154, 158, 174, 179]
sum_ = []

for i in range(len(data_X)) :
    sum_.append(data_Y[i] / 200)

x = np.arange(len(data_X))  # 设定步长
width = 0.4  # 设置数据条宽度
fig, ax = plt.subplots()
p1 = ax.bar(x, data_Y, width)
plt.title("DDPG demo")
plt.xlabel("sheep's v when dog's v is 20")
plt.ylabel("successful escape times in 200 episodes")
ax.set_xticks(x)
ax.set_xticklabels(data_X)

ax2 = ax.twinx()
ax2.plot(data_X, sum_, 'o-', color = 'm')
# 设置纵轴格式
fmt = '%.2f%%'
ax2.set_ylabel("successful escape probability in 200 episodes")

#plt.show()
plt.savefig("DDPG.png")