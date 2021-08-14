import numpy as np
import gym
import time
from rl import DDPG
from env import ChaseEnv

#####################  hyper parameters  ####################
EPISODES = 200
EP_STEPS = 5000
MEMORY_CAPACITY = 10000
RENDER = True

_R = 100
_r = 0
_sheepTheta = 0
_dogTheta = 0
_dogV = 1
_sheepV = 0.8

env = ChaseEnv(_R, _r, _sheepTheta, _sheepV, _dogTheta, _dogV)
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.alpha_bound
a_low_bound = env.alpha_bound[0]
a_high_bound = env.alpha_bound[1]

ddpg = DDPG(a_dim, s_dim, a_bound)
var = 3  # the controller of exploration which will decay during training process
t1 = time.time()
for i in range(EPISODES):
    observation = env.reset()
    ep_r = 0
    for j in range(EP_STEPS):
        if RENDER : env.render()
        # add explorative noise to action
        action = ddpg.choose_action(observation)
        action = np.clip(np.random.normal(action, var), a_low_bound, a_high_bound)
        observation_, reward, done = env.step(action)
        ddpg.store_transition(observation, action, reward, observation_)  # store the transition to memory

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= 0.9995  # decay the exploration controller factor
            ddpg.learn()

        observation = observation_
        ep_r += reward
        if j == EP_STEPS - 1 or done :
            print('Episode: ', i, ' Reward: %i' % (ep_r), 'Explore: %.2f' % var, 'done' if done else '----')
            if ep_r > 5000 : RENDER = True
            break
print('Running time: ', time.time() - t1)