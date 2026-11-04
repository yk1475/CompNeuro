import matplotlib.pyplot as plt

def diagram(ts, ys):
    plt.plot(ts, ys, color="blue", marker="o", markersize=0.1)
    plt.grid(True)
    plt.show()