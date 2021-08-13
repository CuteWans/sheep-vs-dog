import numpy as np
import gym
from gym.envs.classic_control import rendering


class ChaseEnv(object):
    viewer = None
    dt = 0.1
    alpha_bound = [0, np.pi]
    state_dim = 3
    alpha_dim = 1

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

    def step(self, alpha):
        done = False
        r = self.r
        R = self.R
        theta = self.sheepTheta
        dt = self.dt
        v = self.sheepV
        V = self.dogV
        r_ = r + v * np.sin(alpha) * dt
        x = np.sqrt(r ** 2 + (v * dt) ** 2 + 2 * r * v * np.sin(alpha) * dt)
        beta = np.arcsin(v * np.cos(alpha) * dt / x)
        if alpha < np.pi / 2:
            theta -= beta
        else:
            theta += beta
        self.state[0] = r_
        self.state[1] = theta
        if abs(self.state[2] - self.state[1]) <= V / R * dt :
            self.state[2] = self.state[1]
        elif self.state[2] < self.state[1] :
            self.state[2] += V / R * dt
        else:
            self.state[2] -= V / R * dt

        if self.state[0] >= R and np.fabs(self.state[1] - self.state[2]) != 0 :
            done = True

        return done, self.state

    def reset(self):
        self.state[0] = self.r
        self.state[1] = self.sheepTheta
        self.state[2] = self.dogTheta
        return self.state

    def render(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(600, 600)
        circle = rendering.make_circle(100, filled=False)

        circle_transform = rendering.Transform(translation=(300, 300))
        circle.add_attr(circle_transform)
        circle.set_linewidth(5)

        sheep = rendering.make_circle(2)
        sheep_transform = rendering.Transform(translation=(
            self.state[0] * np.cos(self.state[1]) + 300, 300 + self.state[0] * np.sin(self.state[1])))
        sheep.add_attr(sheep_transform)

        self.viewer.add_geom(circle)
        self.viewer.add_geom(sheep)
        self.viewer.render()


if __name__ == "__main__":
    pass
