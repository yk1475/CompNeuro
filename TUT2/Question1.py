import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Tool.plot import diagram

# Initial Conditions
initial_v = -65
initial_u = -1

# Parameters - q1a
I = 10
a = 0.02
b = 0.2
c = -65
d = 8
delta_t = 0.1

# Parameters - q1b
# I = 10
# a = 0.02
# b = 0.25
# c = -65
# d = 2
# delta_t = 0.1

def izhikevich_prime(
    state,
    I,
    a,
    b,
):
    v = state[0]
    u = state[1]

    dv = 0.04*(v**2) + 5*v + 140 - u + I
    du = a * (b*v - u)

    return np.array([
        dv,
        du
    ])


def izhikevich_euler(
    statePrime,
    variable_num,
    initial_values,
    deltaT = delta_t,
    maxT = 20,
    I = I,
    a = a,
    b = b,
    c = c,
    d = d,
):
    ts = np.arange(0, maxT+deltaT, deltaT)
    spikes = 0
    state = np.zeros((variable_num, len(ts)))

    # Initial values setup
    state[:, 0] = initial_values

    for i in range(len(ts)-1):
        current = state[:, i].copy()

        # Spiked last step -> reset before integrating
        if current[0] >= 30:
            current[0] = c
            current[1] += d

        state[:, i+1] = current + deltaT * statePrime(current, I, a, b)

        # Spike this step -> pin to 30 so it shows on the plot as 30
        if state[0, i+1] >= 30:
            state[0, i+1] = 30
            spikes += 1

    return ts, state, spikes * 1000 / maxT

ts, state, freq = izhikevich_euler(izhikevich_prime, 2, [initial_v, initial_u], delta_t, 200)

diagram(ts, state[0])


# Q1c 

delta_I = 0.1
max_I = 20
Is = np.arange(0, max_I+delta_I, delta_I)
freqs = np.zeros((2, len(Is)))

for i in range(len(Is)):
    # Excitatory neuron parameters
    _, _, excitatory_freq = izhikevich_euler(
        izhikevich_prime,
        variable_num=2,
        initial_values=[initial_v, initial_u],
        maxT=1000,
        I=Is[i],
        a=0.02,
        b=0.2,
        c=-65,
        d=8
    )

    # Inhibitory neuron parameters
    _, _, inhibitory_freq = izhikevich_euler(
        izhikevich_prime,
        variable_num=2,
        initial_values=[initial_v, initial_u],
        maxT=1000,
        I=Is[i],
        a=0.02,
        b=0.25,
        c=-65,
        d=2
    )

    freqs[0, i] = excitatory_freq
    freqs[1, i] = inhibitory_freq


# F-I curves: excitatory vs inhibitory
plt.plot(Is, freqs[0, :], label='Excitatory')
plt.plot(Is, freqs[1, :], label='Inhibitory')
plt.xlabel('Current I')
plt.ylabel('Firing rate (Hz)')
plt.legend()
plt.grid(True)
plt.show()
