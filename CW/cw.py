from iznetwork import IzNetwork
import numpy as np
import random
import matplotlib.pyplot as plt

# probabilities
ps = [0, 0.1, 0.2, 0.3, 0.4, 0.5]


inhibitory_nodes = 200 # inhibitory pool (one module)
excitatory_nodes = 100 # excitatory nodes per module
m = 1000 # E->E edges per module
N = 8 # Number of excitatory modules

T = 1000 # simulation length in ms

# extra I current into each neuron following Poisson process with poisson_lmb
poisson_lmb = 0.01 
I = 15

rng = np.random.default_rng() # random gen

def EE(p, n=excitatory_nodes, m=m, N=N):
    # A[i, j] = True means edge i -> j
    A = np.zeros((n*N,n*N))

    for c in range(N):
        start = c * n
        edge_count = 0
        # Randomly link nodes
        while edge_count < m:
            i = random.randrange(start, start+n)
            j = random.randrange(start, start+n)
            if i != j and not A[i, j]:
                A[i, j] = 17
                # A[j, i] = True
                edge_count += 1


    # Rewire with probability p, keep one endpoint and move to random node in E
    src, tgt = np.nonzero(A)

    for i, j in zip(src, tgt):
        if random.random() < p:

            while True:
                k = random.randrange(len(A))
                # no self loop or already existing edge
                if k != i and not A[i, k]:
                    break

            A[i, j] = 0
            A[i, k] = 17

    return A


def II(n=inhibitory_nodes):
    A = rng.uniform(-1, 0, size=(n,n))
    return A
    
def IE(inh=inhibitory_nodes, exc=excitatory_nodes, M=N):
    A = 2 * rng.uniform(-1, 0, size=(inh, exc * M))
    return A

def EI(exc=excitatory_nodes, inh=inhibitory_nodes, M=N):
    A = np.zeros((exc * M, inh))
    inh_per_module = inh//M
    for k in range(inh):
        start_e = (k // inh_per_module) * exc
        sources = rng.choice(np.arange(start_e, start_e + exc), size=4, replace=False)
        A[sources, k] = 50 * rng.uniform(0, 1, size=4)

    return A

def neuron_params(exc=excitatory_nodes, inh=inhibitory_nodes, N=N):
    N_ex = N*exc
    # one r per excitatory neuron
    r_ex = rng.random(N_ex)
    r_in = rng.random(inh)

    a = np.concatenate([0.02*np.ones(N_ex), 0.02 + 0.08 *  r_in])
    b = np.concatenate([0.2*np.ones(N_ex), 0.25 - 0.05 *  r_in])
    c = np.concatenate([-65+15*r_ex**2, -65*np.ones(inh)])
    d = np.concatenate([8-6*r_ex**2, 2*np.ones(inh)])

    return a,b,c,d
    

def build_network(p, exc=excitatory_nodes, inh=inhibitory_nodes, N=N):
    N_ex = exc * N
    N = N_ex + inh

    # Weights
    W = np.zeros((N, N))
    W[:N_ex, :N_ex] = EE(p)
    W[:N_ex, N_ex:] = EI()
    W[N_ex:, :N_ex] = IE()
    W[N_ex:, N_ex:] = II()

    # Delay: E->E 1-20ms and rest 1ms
    D = np.ones((N,N), dtype=int)
    D[:N_ex, :N_ex] = rng.uniform(1, 21, size=(N_ex, N_ex))

    # parameters for exitatory, inhibitory
    a,b,c,d = neuron_params()

    net = IzNetwork(N, Dmax=20)
    net.setWeights(W)
    net.setDelays(D)
    net.setParameters(a,b,c,d)

    return net, W


def simulate(net, total_neurons=N*excitatory_nodes+inhibitory_nodes, T=T):
    fired_times, neurons = [], []
    for t in range(T):
        net.setCurrent(I*rng.poisson(poisson_lmb, total_neurons))
        fired = net.update()
        fired_times.extend([t]*len(fired))
        neurons.extend(fired)
    return np.array(fired_times), np.array(neurons)


def plot_connectivity(Ws, ps):
    n_ex = excitatory_nodes * N
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, W, p in zip(axes.flat, Ws, ps):
        ax.spy(W[:n_ex, :n_ex], markersize=0.1)
        ax.set_title(f"p = {p}")
        ax.set_xlabel("Target neuron")
        ax.set_ylabel("Source neuron")
    fig.tight_layout()
    plt.show()


def firing_rates(times, neurons, n=excitatory_nodes, N=N, T=T, window=50, step=20):
    starts = np.arange(0, T - window + 1, step) # 0, 20, ... -> 48 windows
    modules = neurons // n # module of each spike & (8, 9 = inhibitory)

    rates = np.zeros((N, len(starts)))
    for c in range(N):
        module_times = times[modules == c] # from module c only
        for k, start in enumerate(starts):
            in_window = (module_times >= start) & (module_times < start + window)
            rates[c, k] = in_window.sum() / window # spikes per ms
    return starts, rates


def plot_raster_and_firing_rate(all_spikes, ps, N_ex=excitatory_nodes*N):
    for (times, spiked_neurons), p in zip(all_spikes, ps):
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        fig.suptitle(f"p = {p}")

        # first graph: raster plot
        raster_ax = axes[0]
        cond = spiked_neurons < N_ex
        raster_ax.scatter(times[cond], spiked_neurons[cond], s=2, marker='|')
        raster_ax.set_ylabel("Neuron")

        # second graph: firing rate plot
        fr_ax = axes[1]
        starts, rates = firing_rates(times, spiked_neurons)
        for module in range(N):
            fr_ax.plot(starts, rates[module])
        fr_ax.set_xlabel("Time (ms)")
        fr_ax.set_ylabel("Spikes / ms")
        fr_ax.set_xlim(0, 1000)

        plt.show()


def main():
    nets, Ws = [], []
    for p in ps:
        net, W = build_network(p)
        nets.append(net)
        Ws.append(W)

    plot_connectivity(Ws, ps)

    all_spikes = [simulate(net) for net in nets]
    plot_raster_and_firing_rate(all_spikes, ps)


main()