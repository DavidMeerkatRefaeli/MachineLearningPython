import matplotlib.pyplot as plt

# Plot moving centroids
from ex7.RunKMeans import kMeans


def connectpoints(p1, p2):
    x1, x2 = p1[0], p2[0]
    y1, y2 = p1[1], p2[1]
    plt.plot([x1, x2], [y1, y2], 'k-')


# todo: fix, once I'm a matplotlib ninja
def interactive_plot(X, k, initial_centroids, iterations = 10):
    previous = None
    current = initial_centroids
    # plt.ion()
    for i in range(iterations):
        _ = input("Press any key to continue")
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=idx)
        plt.scatter(current[:, 0], current[:, 1], marker='x', c='k')
        for j in range(k):
            if previous is not None:
                connectpoints(previous[j], current[j])
        plt.show()
        newcenter, idx = kMeans(X, k, current)
        previous = current
        current = newcenter
