import numpy as np

# TODO : clean up for TUT1

# Euler
def euler(
    statePrime,
    variable_num,
    initial_values,
    deltaT = 0.01,
    maxT = 20,
):
    ts = np.arange(0, maxT+deltaT, deltaT)

    state = np.zeros((variable_num, len(ts)))

    # Initial values setup
    state[:, 0] = initial_values

    for i in range(len(ts)-1):
        state[:,i+1] = state[:,i] + deltaT * statePrime(state[:,i])

    return ts, state


def rk4(
    statePrime,
    variable_num,
    initial_values,
    deltaT = 0.01,
    maxT = 20
):
    ts = np.arange(0, maxT+deltaT, deltaT)
    
    state = np.zeros((variable_num, len(ts)))

    # Initial values setup
    state[:, 0] = initial_values

    for i in range(len(ts)-1):
        z = state[:, i]

        k1 = statePrime(z)
        k2 = statePrime(z + deltaT*k1/2)
        k3 = statePrime(z + deltaT*k2/2)
        k4 = statePrime(z + deltaT*k3)

        state[:, i+1] = z + deltaT * 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return ts, state