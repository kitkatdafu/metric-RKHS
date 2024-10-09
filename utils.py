import numpy as cp

def calc_accuracy(y: cp.ndarray, phi: cp.ndarray, M_hat: cp.ndarray, T):
    y_hat = []
    for i in range(T):
        a, b, c = i * 3, i * 3 + 1, i * 3 + 2
        distance = (2 * phi[a] - phi[b] - phi[c]).T @ M_hat @ (phi[c] - phi[b])
        y_hat.append(cp.sign(distance))
    y_hat = cp.array(y_hat)
    return cp.mean(y == y_hat)


def triplets_distance(k: cp.ndarray, G: cp.ndarray, T: int, deterministic: bool):

    y_s = []
    distance_s = []

    for i in range(T):
        h, i, j = i * 3, i * 3 + 1, i * 3 + 2
        distance = (2 * k[h] - k[i] - k[j]).T @ G @ (k[j] - k[i])

        if deterministic:
            y_t = cp.sign(distance)
        else:
            p_t = 1 / (1 + cp.exp(distance))
            y_t = -1 if cp.random.rand() < p_t else 1

        y_s.append(y_t)
        distance_s.append(distance)

    return cp.array(y_s), cp.array(distance_s)


def get_triplets(x: cp.ndarray, T: int):

    triplet_s = []
    for i in range(T):
        h, i, j = i * 3, i * 3 + 1, i * 3 + 2
        triplet = [x[h], x[i], x[j]]
        triplet_s.append(triplet)

    return cp.array(triplet_s)