import numpy as np


def mcmc2d(func, interval, N):
    y = np.random.uniform(interval[0], interval[1])
    x = np.random.uniform(interval[0], y)
    samples = [func(x, y)]
    mc_points = [(x, y)]

    def Omega(X1, X2):
        xa1, yb1 = X1
        xa2, yb2 = X2
        return 1 / yb2

    for i in range(N):
        y = np.random.uniform(interval[0], interval[1])
        x = np.random.uniform(interval[0], y)

        new_sample = func(x, y)
        ratio = new_sample / samples[i] * Omega((x, y), mc_points[i]) / Omega(mc_points[i], (x, y))

        r = np.random.uniform(0, 1)

        if r < ratio:
            samples.append(new_sample)
            mc_points.append((x, y))
        else:
            samples.append(samples[i])
            mc_points.append(mc_points[i])

    return np.array(samples)


def mcmc1d(func, interval, N):
    x = np.random.uniform(interval[0], interval[1])
    d = interval[1] - interval[0]
    samples = [func(x)]
    mc_points = [x]

    def Omega(X1, X2):
        return 1 / d

    for i in range(N):
        x = np.random.uniform(interval[0], interval[1])

        new_sample = func(x)
        ratio = new_sample / samples[i] * Omega(x, mc_points[i]) / Omega(mc_points[i], x)

        r = np.random.uniform(0, 1)

        if r < ratio:
            samples.append(new_sample)
            mc_points.append(x)
        else:
            samples.append(samples[i])
            mc_points.append(mc_points[i])

    return np.array(samples)


def mcmc_time_ordered(func, dim, interval, N, burn_in=1000):
    t_max = interval[1]
    t_min = interval[0]

    def generate_t():
        t_list = np.array([t_max] * (dim + 1), dtype=np.float64)
        for i in range(1, dim + 1):
            t_list[i] = np.random.uniform(t_min, t_list[i - 1])

        t_list = t_list[1:]
        return t_list

    mc_points = [generate_t()]
    samples = [np.abs(func(*(mc_points[-1])))]

    def omega(X1, X2):
        return 1 / np.prod(X2[:-1] - t_min)

    for i in range(N):
        t_list = generate_t()

        new_sample = np.abs(func(*t_list))
        ratio = new_sample / samples[i] * omega(t_list, mc_points[i]) / omega(mc_points[i], t_list)
        ratio = min(ratio, 1)

        r = np.random.uniform(0, 1)

        if r < ratio:
            samples.append(new_sample)
            mc_points.append(t_list)
        else:
            samples.append(samples[i])
            mc_points.append(mc_points[i])
    nominator = np.array([np.exp(1j*np.angle(func(*p))) for p in mc_points])
    samples = np.array(samples) / np.math.factorial(dim) * (t_max - t_min) ** dim
    return np.mean(nominator) / np.mean(1/samples[burn_in:]), nominator, samples



if __name__ == "__main__":
    def f9(x, y, z, a, b, c, d, e, f):
        return np.exp(-(x + y + z + a + b + c + d + e + f) ** 2)


    def f4(x, y, z, a):
        return np.exp(-(x + y + z + a) ** 2)


    def f5(x, y, z, a, b):
        return np.exp(-(x + y + z + a + b) ** 2)


    def f2(x, y):
        return np.exp(-(x) ** 2)


    N = 500000
    _, a = mcmc_time_ordered(f5, 5, [0, 1], N=N)
    print("finished")
    inv_samples = 1 / a

    mean_list = []
    start = 1000
    for i in range(N - 100000, N):
        mean = 1 / np.mean(inv_samples[start:i])
        mean_list.append(mean)

    import matplotlib.pyplot as plt

    plt.plot(mean_list)
    plt.plot([0.000199348] * len(mean_list))


    plt.savefig('mcmc_integral.png', dpi=300)








