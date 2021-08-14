import numpy as np
import gym
from gym.envs.classic_control import rendering


class ChaseEnv(object):
    viewer = None
    dt = 0.1
    alpha_bound = [0, np.pi]
    state_dim = 2
    action_dim = 1

    def __init__(self, _R, _r, _sheepTheta, _sheepV, _dogTheta, _dogV):
        self.state = np.zeros(3, dtype=np.float32)
        self.state[0] = _r
        self.state[1] = _sheepTheta
        self.state[2] = _dogTheta

        self.R = _R
        self.sheepV = _sheepV
        self.dogV = _dogV

        self.r = _r
        self.sheepTheta = _sheepTheta
        self.dogTheta = _dogTheta

    def step(self, action):
        done = False
        r = self.state[0]
        R = self.R
        theta = self.state[1]
        dt = self.dt
        v = self.sheepV
        V = self.dogV
        r_ = r + v * np.sin(action) * dt
        x = np.sqrt(r ** 2 + (v * dt) ** 2 + 2 * r * v * np.sin(action) * dt)
        beta = np.arcsin(v * np.cos(action) * dt / x)
        if action < np.pi / 2 :
            theta -= beta
        else :
            theta += beta
        self.state[0] = r_
        self.state[1] = theta
        self.state[1] -= int(self.state[1] / (2 * np.pi)) * (2 * np.pi)
        theta = np.fabs(self.state[1] - self.state[2])
        if theta > np.pi : theta = 2 * np.pi - theta
        #if theta <= V / R * dt:
        #    self.state[2] = self.state[1]
        #el
        if self.state[2] > ((self.state[1] + np.pi) - 2 * np.pi if (self.state[1] + np.pi) > 2 * np.pi else (self.state[1] + np.pi)) :
            self.state[2] += V / R * dt
        else:
            self.state[2] += 2 * np.pi - V / R * dt
        self.state[2] -= int(self.state[2] / (2 * np.pi)) * (2 * np.pi)

        if self.state[0] >= R and np.fabs(self.state[1] - self.state[2]) != 0:
            done = True

        #k, a, b, c, d = 1, 1, 0, 1, 0
        #reward = k * (a * theta + b) / (c * (self.R - r_) + d)
        a, b, c = 1, 1, -1
        reward = a * r_ + b * theta + c

        #print(r_, theta)

        return [self.state[0], theta], reward[0], done

    def reset(self):
        self.state[0] = self.r
        self.state[1] = self.sheepTheta
        self.state[2] = self.dogTheta
        if self.viewer is not None : del self.viewer
        return [self.state[0], np.fabs(self.state[1] - self.state[2])]

    def render(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(600, 600)
        circle = rendering.make_circle(self.R, filled=False)

        circle_transform = rendering.Transform(translation=(300, 300))
        circle.add_attr(circle_transform)
        circle.set_linewidth(5)

        sheep = rendering.make_circle(2)
        sheep_transform = rendering.Transform(translation=(
            self.state[0] * np.cos(self.state[1]) + 300, 300 + self.state[0] * np.sin(self.state[1])))
        sheep.add_attr(sheep_transform)

        dog = rendering.make_circle(4)
        dog.set_color(.7, .5, .5)
        dog_transform = rendering.Transform(translation=(
            self.R * np.cos(self.state[2]) + 300, 300 + self.R * np.sin(self.state[2])))
        dog.add_attr(dog_transform)

        self.viewer.add_geom(circle)
        self.viewer.add_geom(sheep)
        self.viewer.add_geom(dog)
        self.viewer.render()
