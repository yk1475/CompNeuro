import random
import numpy as np
import matplotlib.pyplot as plt

class IzhikevichNeurons:

    # N: number of neurons
    # I: current
    # isExcitatory: true = excitatory, false = inhibitory
    def __init__(self, N, I, isExcitatory):
        self.N = N
        self.I = I
        self.isExcitatory = isExcitatory

        self.a = np.zeros(N)
        self.b = np.zeros(N)
        self.c = np.zeros(N)
        self.d = np.zeros(N)
        self.setParameters()
        self.setConnections()

        self.maxT = 200
        self.deltaT = 0.1

        # initial values
        self.initial_v = -65
        self.initial_u = -1

    def setParameters(self):
        for i in range(self.N):
            r = random.random()

            if self.isExcitatory:
                self.a[i] = 0.02
                self.b[i] = 0.2
                self.c[i] = -65 + 15*(r**2)
                self.d[i] = 8 - 6*(r**2)
            else:
                self.a[i] = 0.02 + 0.08*r
                self.b[i] = 0.25 - 0.05*r
                self.c[i] = -65
                self.d[i] = 2

    def setDelta(self, deltaT):
        self.deltaT = deltaT

    def setMaxTime(self, maxT):
        self.maxT = maxT

    def euler(self):
        ts = np.arange(0, self.maxT + self.deltaT, self.deltaT)
        us = np.zeros((self.N, len(ts)))
        vs = np.zeros((self.N, len(ts)))
    
        # Initial values setup
        us[:, 0] = self.initial_u
        vs[:, 0] = self.initial_v

        # internal_I[j, step]: synaptic current arriving at neuron j at that step
        internal_I = np.zeros((self.N, len(ts)))
    
        for step in range(len(ts)-1):
            v = vs[:, step].copy()
            u = us[:, step].copy()
    
            # Spiked last step -> reset before integrating
            fired = v >= 30
            v[fired] = self.c[fired]
            u[fired] += self.d[fired]

            # from neuron i (if fired)
            for i in np.where(fired)[0]:
                # check every j neuron
                for j in range(self.N):
                    arriveT = step + self.D[i, j]
                    if arriveT < len(ts):
                        internal_I[j, arriveT] += self.W[i, j]
            
    
            # Izhikevich derivatives
            dv = 0.04*(v**2) + 5*v + 140 - u + self.I + internal_I[:, step]
            du = self.a * (self.b*v - u)

            vs[:, step+1] = v + self.deltaT * dv
            us[:, step+1] = u + self.deltaT * du
    
            # Neurons that spiked this step -> pin to 30
            vs[vs[:, step+1] >= 30, step+1] = 30
    
        return ts, vs, us

    # Q2c spec: D(i, j) = 5, W(i, j) = 50 / sqrt(K)
    def setConnections(self, connected=None, delay=5, weight=50):
        if connected is None:
            connected = ~np.eye(self.N, dtype=bool)

        # K[j]: number of incoming synapses of post-synaptic neuron j
        K = connected.sum(axis=0)

        # dividing by sqrt(K) (shape (N,)) applies K[j] down each column j
        self.W = np.where(connected, weight / np.sqrt(np.maximum(K, 1)), 0)
        self.D = np.full((self.N, self.N), delay, dtype=int)


neurons = IzhikevichNeurons(N=100, I=5, isExcitatory=True)
neurons.setMaxTime(1000)
neurons.setConnections(connected=np.zeros((100, 100), dtype=bool))  # Q2b: no connections
ts, vs, us = neurons.euler()

neuron_idx, t_idx = np.where(vs >= 30)
plt.scatter(ts[t_idx], neuron_idx, s=2, marker='|')
plt.grid(True)
plt.show()