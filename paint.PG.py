import numpy as np
import matplotlib.pyplot as plt

data_X = ['11', '12', '13', '14', '15', '16']
data_Y = [32, 73, 104, 133, 150, 159]
data_Y1 = [32, 50, 80, 114, 133, 155]
data_Y2 = [17, 44, 66, 100, 124, 145]
data_Y3 = [11, 26, 56, 79, 92, 127]
data_Y4 = [7, 22, 44, 55, 89, 108]
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
plt.title("PG demo")
plt.xlabel("sheep's v when dog's v is 20")
plt.ylabel("successful escape times every 200 episodes")
ax.set_xticks(x)
ax.set_xticklabels(data_X)

ax2 = ax.twinx()
ax2.plot(data_X, sum_, 'o-', color = 'm')
# 设置纵轴格式
fmt = '%.2f%%'
ax2.set_ylabel("successful escape probability every 200 episodes")

#plt.show()
plt.savefig("PG.png")