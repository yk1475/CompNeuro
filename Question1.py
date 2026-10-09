import math
import numpy as np
import matplotlib.pyplot as plt
import time

# Q1

def yprime(t):
    return math.exp(t)

def eulerForY(tMax, delta):
    ts = np.arange(0, tMax+delta, delta)
    ys = np.zeros(len(ts))
    ys[0] = 1
    for i in range(len(ys)-1):
        ys[i+1] = ys[i] + delta * yprime(ts[i])

    return ts, ys

def q1a():
    deltas = [0.1, 0.2, 0.5, 1, 2]
    colours = ["blue", "green", "orange", "purple", "black"]
    for i in range(len(deltas)):
        ts, ys = eulerForY(10, deltas[i])
        plt.plot(ts, ys, color=colours[i], marker="o",
            markersize=3, label=f"step {deltas[i]}")
    y2 = np.exp(ts)
    plt.plot(ts, y2, color="red", label="math exp")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title(f"euler")
    plt.grid(True)
    plt.show()


# Q1b
def eulerOscillator(
    c = 0.1,
    m = 1,
    k = 1,
    tMax = 100,
    delta = 0.01,
):

    ts = np.arange(0, tMax+delta, delta)
    # state = [(y,p)]
    state = np.zeros((2, len(ts)))
    state[0,0] = 1
    state[1,0] = 0
    for i in range(len(ts)-1):
        y, p = state[:,i]
        state[1, i+1] = p + delta * ((-c * p - k * y)/m)
        state[0, i+1] = y + delta * p

    return ts, state

def q1b():
    ts, state = eulerOscillator()

    plt.plot(ts, state[0,:], color="blue", marker="o", markersize=0.1)
    plt.grid(True)
    plt.show()


# Q1c
c = 0.1
m = 1
k = 1

def zPrime(z):
    y = z[0]
    p = z[1]

    return np.array([
        p,
        (-c*p - k*y)/m
    ])

def rkOscillator(
        tMax = 100,
        delta = 0.5
):    
    ts = np.arange(0, tMax+delta, delta)
    # [(y, p)]
    zs = np.zeros((2, len(ts)))
    zs[0, 0] = 1
    zs[1, 0] = 0

    for i in range(len(ts)-1):
        z = zs[:, i]

        k1 = zPrime(z)
        k2 = zPrime(z + delta*k1/2)
        k3 = zPrime(z + delta*k2/2)
        k4 = zPrime(z + delta*k3)

        zs[:, i+1] = z + delta * 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return ts, zs


def q1c():
    ts, zs = rkOscillator()

    plt.plot(ts, zs[0:,], color="blue", marker="o", markersize=0.1)
    plt.grid(True)
    plt.show()


