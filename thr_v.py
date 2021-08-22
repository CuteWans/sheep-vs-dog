import numpy as np
import matplotlib.pyplot as plt

data_X = ['11', '12', '13', '14', '15', '16']
data_Y = [32, 73, 104, 133, 150, 159]
data_Y1 = [32, 50, 80, 114, 133, 155]
data_Y2 = [17, 44, 66, 100, 124, 145]
data_Y3 = [11, 26, 56, 79, 92, 127]
data_Y4 = [7, 22, 44, 55, 89, 108]

data_Y_1 = [35, 68, 82, 113, 151, 159]
data_Y1_1 = [53, 61, 95, 110, 166, 176]
data_Y2_1 = [51, 64, 113, 114, 155, 161]
data_Y3_1 = [42, 57, 81, 115, 149, 157]
data_Y4_1 = [24, 46, 77, 102, 124, 156]

data_Y_2 = [130, 132, 154, 158, 174, 179]

sum_PG = []
sum_A3C = []
sum_DDPG = []

for i in range(len(data_X)) :
    sum_PG.append((data_Y[i] + data_Y4[i] + data_Y3[i] + data_Y2[i] + data_Y1[i]) / 1000)
    sum_A3C.append((data_Y_1[i] + data_Y4_1[i] + data_Y3_1[i] + data_Y2_1[i] + data_Y1_1[i]) / 1000)
    sum_DDPG.append(data_Y_2[i] / 200)

fig, ax = plt.subplots()
plt.title("Compare demo")
plt.xlabel("sheep's v when dog's v is 20")
plt.ylabel("successful escape times every 200 episodes")

ax.plot(data_X, sum_PG, 'o-', color = 'g', label = "PG")
ax.plot(data_X, sum_A3C, 'o-', color = 'b', label = "A3C")
ax.plot(data_X, sum_DDPG, 'o-', color = 'r', label = "DDPG")
# 设置纵轴格式
fmt = '%.2f%%'
ax.set_ylabel("successful escape probability every 200 episodes")
plt.legend()
#plt.show()
plt.savefig("Compare_v.png")