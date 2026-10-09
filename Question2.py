import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Earth based pendulum assumption
l = m = 1
g = 9.8


# initial conditions
theta1 = 120
theta2 = -10

p1 = p2 = 0


# Arguments
deltaT = 0.01
tMax = 5


# theta:(theta1, theta2), p:(p1, p2) -> theta prime
def thetaPrime(theta, p):
    theta1 = theta[0]
    theta2 = theta[1]
    p1 = p[0]
    p2 = p[1]

    theta1prime = 6/m*(l**2) * (
        (2*p1 - 3 * np.cos(theta1-theta2) * p2)/
        (16 - 9 * (np.cos(theta1-theta2))**2)
    )

    theta2prime = 6/m*(l**2) * (
            (8*p2 - 3 * np.cos(theta1-theta2) * p1)/
            (16 - 9 * (np.cos(theta1-theta2))**2)
        )

    return np.array([
        theta1prime,
        theta2prime
    ])

# theta:(theta1, theta2), p:(p1, p2) -> p prime
def pPrime(theta, p):
    theta1 = theta[0]
    theta2 = theta[1]

    thetaPrimes = thetaPrime(theta, p)
    theta1Prime = thetaPrimes[0]
    theta2Prime = thetaPrimes[1]

    p1Prime = -1/2 * m * (l**2) * (
        theta1Prime * theta2Prime * np.sin(theta1-theta2) +
        3 * g / l * np.sin(theta1)
    )

    p2Prime = -1/2 * m * (l**2) * (
        -theta1Prime * theta2Prime * np.sin(theta1-theta2) +
        g / l * np.sin(theta2)
    )

    return np.array([
        p1Prime,
        p2Prime
    ])


def euler(
        deltaT=deltaT,
        tMax=tMax,
        theta1Init = theta1,
        theta2Init = theta2,
        p1Init = p1,
        p2Init = p2,
        ):
    ts = np.arange(0, tMax+deltaT, deltaT)
    thetas = np.zeros((2, len(ts)))
    thetas[0,0] = theta1Init
    thetas[1,0] = theta2Init
    ps = np.zeros((2, len(ts)))
    ps[0,0] = p1Init
    ps[1,0] = p2Init

    for i in range(len(ts)-1):
        thetas[:,i+1] = thetas[:,i] + deltaT * thetaPrime(thetas[:,i], ps[:,i])
        ps[:,i+1] = ps[:,i] + deltaT * pPrime(thetas[:,i], ps[:,i])

    return ts, thetas, ps

# theta, p -> theta prime, p prime
def f(theta, p):
    return (thetaPrime(theta, p), pPrime(theta, p))


def rungeKutta(
        deltaT=deltaT,
        tMax=tMax,
        theta1Init = theta1,
        theta2Init = theta2,
        p1Init = p1,
        p2Init = p2,
        ):
    ts = np.arange(0, tMax+deltaT, deltaT)
    thetas = np.zeros((2, len(ts)))
    thetas[0,0] = theta1Init
    thetas[1,0] = theta2Init
    ps = np.zeros((2, len(ts)))
    ps[0,0] = p1Init
    ps[1,0] = p2Init

    for i in range(len(ts)-1):
        k1 = f(thetas[:,i], ps[:,i])
        k2 = f(thetas[:,i] + deltaT/2 * k1[0], ps[:,i] + deltaT/2 * k1[1])
        k3 = f(thetas[:,i] + deltaT/2 * k2[0], ps[:,i] + deltaT/2 * k2[1])
        k4 = f(thetas[:,i] + deltaT * k3[0], ps[:,i] + deltaT * k3[1])

        thetas[:,i+1] = thetas[:,i] + deltaT/6 * (k1[0]+2*k2[0]+2*k3[0]+k4[0])
        ps[:,i+1] = ps[:,i] + deltaT/6 * (k1[1]+2*k2[1]+2*k3[1]+k4[1])

    return ts, thetas, ps


# ts: (N,), thetas: (2, N) -> animation of the pendulum over time
def animate(ts, thetas):
    x1 = l * np.sin(thetas[0])
    y1 = -l * np.cos(thetas[0])
    x2 = x1 + l * np.sin(thetas[1])
    y2 = y1 - l * np.cos(thetas[1])

    fig, ax = plt.subplots()
    ax.set(xlim=(-2.2*l, 2.2*l), ylim=(-2.2*l, 2.2*l), aspect='equal')
    line, = ax.plot([], [], 'o-')

    def update(i):
        line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        ax.set_title(f't = {ts[i]:.2f} s')

    anim = FuncAnimation(fig, update, frames=len(ts), interval=tMax)
    plt.show()


def q2a():
    ts, thetas, _ = rungeKutta()
    animate(ts, thetas)

q2a()