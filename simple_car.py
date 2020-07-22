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
    car = SimpleCar()
    print(car.x)
    u = [
        -1., #2. * np.random.rand() - 1.,
        (np.pi / 4.) * (2. * np.random.rand() - 1.) 
    ]
    for i in range(1000):
        car.render()
        u[1] = np.minimum(np.maximum(u[1] + .1 * (np.pi / 4.) * (2. * np.random.rand() - 1.), -np.pi / 4.), np.pi / 4.)
        car.step(u)
        print(car.x)



class SimpleCar():
    def __init__(self):
        # car's length
        self.l = 3.

        # initial state
        self.x = np.array([
            # rear axle center x-position
            np.random.randn(),
            # rear axle center y-position
            np.random.randn(),
            # direction of the car
            2. * np.pi * np.random.rand()
        ])

        # solver timestep
        self.dt = .01

        self.fig = None
        self.ax = None

    def step(self, u):
        s, phi = u
        assert -1 <= s <= 1
        assert -np.pi / 4. <= phi <= np.pi / 4.
        x, y, theta = self.x
        dxdt = s * np.cos(theta)
        dydt = s * np.sin(theta)
        dthetadt = s * np.tan(phi) / self.l

        dxdt = np.array([dxdt, dydt, dthetadt])
        self.x += dxdt * self.dt
        return self.x

    def render(self):
        if self.fig is None:
            assert self.ax is None
            self.fig, self.ax = plt.subplots()
        if not self.ax.lines:
            self.ax.plot([], [])
            self.ax.plot([], [])
            self.ax.grid()
            self.ax.set_xlim([-10, 10])
            self.ax.set_ylim([-10, 10])
        ax, rect = self.ax.lines

        xs = np.zeros((2, ), np.float_)
        ys = np.zeros((2, ), np.float_)
        xs[1], ys[1], theta = self.x
        xs[0] = xs[1] + np.cos(theta) * self.l
        ys[0] = ys[1] + np.sin(theta) * self.l
        ax.set_data(xs, ys)

        tmp = .333
        rxs = [
            xs[0] + tmp * self.l * np.cos(theta + np.pi / 4.),
            xs[0] + tmp * self.l * np.cos(theta - np.pi / 4.),
            xs[1] - tmp * self.l * np.cos(theta + np.pi / 4.),
            xs[1] - tmp * self.l * np.cos(theta - np.pi / 4.),
            xs[0] + tmp * self.l * np.cos(theta + np.pi / 4.),
        ]
        rys = [
            ys[0] + tmp * self.l * np.sin(theta + np.pi / 4.),
            ys[0] + tmp * self.l * np.sin(theta - np.pi / 4.),
            ys[1] - tmp * self.l * np.sin(theta + np.pi / 4.),
            ys[1] - tmp * self.l * np.sin(theta - np.pi / 4.),
            ys[0] + tmp * self.l * np.sin(theta + np.pi / 4.),
        ]
        rect.set_data(rxs, rys)

        # self.ax.relim()
        # self.ax.autoscale_view()
        plt.draw()
        plt.pause(1e-07)
    


if __name__ == "__main__":
    main()
