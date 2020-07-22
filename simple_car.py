#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os

import numpy as np
import matplotlib.pyplot as plt

# __TensorFlow logging level__
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# import tensorflow as tf
# from tensorflow import keras


def main():
    env = SimpleCar()
    while True:
        observation, done = env.reset(), False
        action = [
            np.random.choice([-1., 1]), #1., #2. * np.random.rand() - 1.,
            (np.pi / 4.) * (2. * np.random.rand() - 1.) 
        ]
        for i in range(512):
            env.render()
            action[1] = np.minimum(np.maximum(action[1] + .1 * (np.pi / 4.) * (2. * np.random.rand() - 1.), -np.pi / 4.), np.pi / 4.)
            # action[1] = (np.pi / 4.) * (2. * np.random.rand() - 1.)
            new_observation, reward, done, info = env.step(action)
            print(reward)
            if done:
                break
            observation = new_observation
    # plt.show()


def rotation_matrix(alpha):
    return np.array([
        [np.cos(alpha), np.sin(alpha)],
        [-np.sin(alpha), np.cos(alpha)]
    ])


class SimpleCar():
    def __init__(self):
        # solver timestep
        self.dt = .05

        self.fig = None
        self.ax = None

        self.xlim = [-15, 15]
        self.ylim = [-2.5, 27.5]

        self.a = 2.815
        self.b = 1.054
        self.c = 0.910
        self.d = self.a + self.b + self.c
        self.e = 1.586
        self.f = 2.096
        self.g = 1.557
        self.h = 1.860
        self.j = 0.5

        self.scalex = 2.
        self.scaley = 1.5

    def reset(self):
        self.u = np.array([0., np.pi / 4.])

        # initial state
        self.x = np.array([
            # rear axle center x-position
            20. * np.random.rand() - 10.,
            # rear axle center y-position
            10. * np.random.rand() + 10.,
            # direction of the car
            2. * np.pi * np.random.rand()
        ])

        observation = self.x
        return observation

    def step(self, u):
        s, phi = u
        assert -1 <= s <= 1
        assert -np.pi / 4. <= phi <= np.pi / 4.
        self.u = u

        x, y, theta = self.x
        dxdt = s * np.cos(theta)
        dydt = s * np.sin(theta)
        dthetadt = s * np.tan(phi) / self.a

        dxdt = np.array([dxdt, dydt, dthetadt])
        self.x += dxdt * self.dt

        tmp = self._bbox()
        x, y, theta = self.x
        # rotate
        tmp = np.dot(tmp, rotation_matrix(theta))
        # translate
        tmp += np.array([[x, y]])

        observation = self.x
        reward = 1. / np.sqrt(self.x[0] ** 2 + self.x[1] ** 2)
        done = (self._border_fcn(tmp[:, 0]) > tmp[:, 1]).any() \
                or (tmp[:, 0] < self.xlim[0]).any() \
                or (tmp[:, 0] > self.xlim[1]).any() \
                or (tmp[:, 1] < self.ylim[0]).any() \
                or (tmp[:, 1] > self.ylim[1]).any()
        info = dict()
        return observation, reward, done, info

    def _bbox(self):
        bbox = np.array([
            [self.a + self.b,  .5 * self.f],
            [self.a + self.b, -.5 * self.f],
            [        -self.c, -.5 * self.f],
            [        -self.c,  .5 * self.f]
        ])
        return bbox

    def _rear_wheel(self):
        wheel = np.array([
            [-self.j, 0.],
            [ self.j, 0.]
        ])
        return wheel

    def _front_wheel(self):
        wheel = self._rear_wheel()
        s, phi = self.u
        wheel = np.dot(wheel, rotation_matrix(phi))
        return wheel

    def _parking_place(self):
        place = np.array([
            [                      -15.,                    0.],
            [-self.scalex * .5 * self.f,                    0.],
            [-self.scalex * .5 * self.f, -self.scaley * self.d],
            [ self.scalex * .5 * self.f, -self.scaley * self.d],
            [ self.scalex * .5 * self.f,                    0.],
            [                       15.,                    0.],
        ])
        place += np.array([[0., 5.]])
        return place

    def _border_fcn(self, x):
        return -self.scaley * self.d * (np.sign(x + self.scalex * .5 * self.f) - np.sign(x - self.scalex * .5 * self.f)) / 2. + 5.

    def render(self):
        if self.fig is None:
            assert self.ax is None
            self.fig, self.ax = plt.subplots()
        if not self.ax.lines:
            self.ax.plot([], [], "C0", linewidth=3)
            for _ in range(4):
                self.ax.plot([], [], "C1", linewidth=3)
            self.ax.plot([], [], "C2o", markersize=6)

            # self.ax.plot(*self._parking_place().T.tolist(), "C3", linewidth=3)

            x = np.linspace(-15, 15, 1000)
            y = self._border_fcn(x)
            self.ax.plot(x, y, "C3", linewidth=3)
            self.ax.plot([0], [0], "C3o", markersize=6)

            self.ax.grid()
            self.ax.set_xlim(self.xlim)
            self.ax.set_aspect("equal")
            self.ax.set_ylim(self.ylim)
        bbox, lfw, rfw, lrw, rrw, center = self.ax.lines[:6]

        tmp = self._bbox()
        x, y, theta = self.x
        # rotate
        tmp = np.dot(tmp, rotation_matrix(theta))
        # translate
        tmp += np.array([[x, y]])
        # repeat 1st point (to close the drawed object)
        tmp = np.concatenate([tmp, tmp[[0]]])
        bbox.set_data(tmp.T)

        tmp = self._front_wheel()
        tmp += np.array([[self.a,  self.e / 2.]])
        # rotate
        tmp = np.dot(tmp, rotation_matrix(theta))
        # translate
        tmp += np.array([[x, y]])
        lfw.set_data(tmp.T)

        tmp = self._front_wheel()
        tmp += np.array([[self.a, -self.e / 2.]])
        # rotate
        tmp = np.dot(tmp, rotation_matrix(theta))
        # translate
        tmp += np.array([[x, y]])
        rfw.set_data(tmp.T)

        tmp = self._rear_wheel()
        tmp += np.array([[0.,  self.g / 2.]])
        # rotate
        tmp = np.dot(tmp, rotation_matrix(theta))
        # translate
        tmp += np.array([[x, y]])
        lrw.set_data(tmp.T)

        tmp = self._rear_wheel()
        tmp += np.array([[0., -self.g / 2.]])
        # rotate
        tmp = np.dot(tmp, rotation_matrix(theta))
        # translate
        tmp += np.array([[x, y]])
        rrw.set_data(tmp.T)

        center.set_data([x], [y])

        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(1e-07)
    

if __name__ == "__main__":
    main()
