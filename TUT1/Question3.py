import numpy as np
import matplotlib.pyplot as plt

I = 10

gNa = 120
gK = 36
gL = 0.3
ENa = 115
EK = -12
EL = 10.6

# initial values
v = 0
m = 0
n = 0
h = 0

# state: (v, m, n, h)
def statePrime(state):
    v = state[0]
    m = state[1]
    n = state[2]
    h = state[3]

    mPrime = (
        (2.5-0.1*v)/(np.exp(2.5-0.1*v)-1) * (1-m)
        - 4*np.exp(-v/18) * m
    )
    nPrime = (
        (0.1-0.01*v)/(np.exp(1-0.1*v)-1) * (1-n)
        - 0.125*np.exp(-v/80) * n
    )
    hPrime = (
        (0.07*np.exp(-v/20)) * (1-h)
        - 1/(np.exp(3-0.1*v)+1) * h
    )
    vPrime = (
        - gNa * (m**3) * h * (v-ENa)
        - gK * (n**4) * (v-EK)
        - gL * (v-EL)
        + I
    )

    return np.array([
        vPrime,
        mPrime,
        nPrime,
        hPrime,
    ])


def euler(
    deltaT = 0.01,
    maxT = 20
):
    ts = np.arange(0, maxT+deltaT, deltaT)

    # state: (v, m, n, h)
    state = np.zeros((4, len(ts)))

    # Initial values setup
    state[0,0]=v
    state[1,0]=m
    state[2,0]=n
    state[3,0]=h

    for i in range(len(ts)-1):
        state[:,i+1] = state[:,i] + deltaT * statePrime(state[:,i])

    return ts, state

def rk4(
    deltaT = 0.01,
    maxT = 20
):
    ts = np.arange(0, maxT+deltaT, deltaT)
    
    # state: (v, m, n, h)
    state = np.zeros((4, len(ts)))

    # Initial values setup
    state[0,0]=v
    state[1,0]=m
    state[2,0]=n
    state[3,0]=h

    for i in range(len(ts)-1):
        z = state[:, i]

        k1 = statePrime(z)
        k2 = statePrime(z + deltaT*k1/2)
        k3 = statePrime(z + deltaT*k2/2)
        k4 = statePrime(z + deltaT*k3)

        state[:, i+1] = z + deltaT * 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return ts, state

def plot(ts, vs):
    plt.plot(ts, vs, color="blue", marker="o", markersize=0.1)
    plt.grid(True)
    plt.show()


ts, state = rk4(deltaT=0.01, maxT=100)
plot(ts, state[0,:])

