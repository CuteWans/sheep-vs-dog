import numpy as np
import gym
import time
from rl import Actor
from rl import Critic
from env import ChaseEnv
import tensorflow as tf

#####################  hyper parameters  ####################
EPISODES = 1000
EP_STEPS = 800
MEMORY_CAPACITY = 10000
RENDER = True

_R = 100
_r = 20
_sheepTheta = 0
_dogTheta = 0
_dogV = 20
_sheepV = 11

env = ChaseEnv(_R, _r, _sheepTheta, _sheepV, _dogTheta, _dogV)
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.alpha_bound
a_low_bound = env.alpha_bound[0]
a_high_bound = env.alpha_bound[1]

sess = tf.Session()

LR_A = 0.001
LR_C = 0.01
actor = Actor(sess, n_features = s_dim, lr = LR_A, action_bound = [0, np.pi])
critic = Critic(sess, n_features = s_dim, lr = LR_C)

sess.run(tf.global_variables_initializer())

tot = 0
var = 3  # the controller of exploration which will decay during training process
t1 = time.time()
for i in range(EPISODES):
    observation = env.reset()
    ep_r = 0
    ep_rs = []
    for j in range(EP_STEPS):
        #if RENDER : env.render()
        # add explorative noise to action
        action = actor.choose_action(observation)
        action = np.clip(np.random.normal(action, var), a_low_bound, a_high_bound)
        observation_, reward, done = env.step(action)
        td_error = critic.learn(observation, reward, observation_)
        actor.learn(observation, action, td_error)
        
        observation = observation_
        ep_rs.append(reward)
        
        ep_r += reward
        
        if j == EP_STEPS - 1 or observation[0] >= _R or done :
            if ep_r == 1000 : tot += 1
            if i == 199 or i == 399 or i == 599 or i == 799 or i == 999 :
                print("Epi: ", i, "successful: ", tot)
                tot = 0
            print('Episode: ', i, ' Reward: %i' % (ep_r), 'Explore: %.2f' % var, 'done' if done else '----')
            if ep_r > 0 and ep_r < 100 : RENDER = True
            break
print('Running time: ', time.time() - t1)