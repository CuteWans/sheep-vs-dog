import numpy as np
import matplotlib.pyplot as plt

data_X = ['11', '12', '13', '14', '15', '16']
data_Y = [35, 68, 82, 113, 151, 159]
data_Y1 = [53, 61, 95, 110, 166, 176]
data_Y2 = [51, 64, 113, 114, 155, 161]
data_Y3 = [42, 57, 81, 115, 149, 157]
data_Y4 = [24, 46, 77, 102, 124, 156]
sum_ = []

for i in range(len(data_X)) :
    sum_.append((data_Y[i] + data_Y4[i] + data_Y3[i] + data_Y2[i] + data_Y1[i]) / 1000)

x = np.arange(len(data_X))  # 设定步长
width = 0.1  # 设置数据条宽度
fig, ax = plt.subplots()
p1 = ax.bar(x - width * 2, data_Y, width)
p2 = ax.bar(x - width * 1, data_Y1, width)
p3 = ax.bar(x, data_Y2, width)
p4 = ax.bar(x + width, data_Y3, width)
p5 = ax.bar(x + width * 2, data_Y4, width)
plt.title("A3C demo")
plt.xlabel("sheep's v when dog's v is 20")
plt.ylabel("successful escape times every 200 episodes")
ax.set_xticks(x)
ax.set_xticklabels(data_X)

ax2 = ax.twinx()
ax2.plot(data_X, sum_, 'o-')
# 设置纵轴格式
fmt = '%.2f%%'
ax2.set_ylabel("successful escape probability every 200 episodes")

#plt.show()
plt.savefig("A3C.png")