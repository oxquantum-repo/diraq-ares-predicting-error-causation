import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

def gaussian(x, mu, std):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / std) ** 2)

@dataclass
class Distribution:
    p_init_0: float = 0.5
    tau_m: float = 4
    T1:float = 40
    mu_0:float = 0
    mu_1:float = 1
    sigma_0:float = 0.1
    sigma_1:float = 0.1

    def __post_init__(self):
        self.p_init_1 = 1 - self.p_init_0
        self.p_no_decay = self.p_init_1 * np.exp(-self.tau_m / self.T1)
        self.p_decay = 1 - self.p_no_decay

    def probability(self, I):
        ground = self.p_init_0 * gaussian(I, self.mu_0, self.sigma_0)
        excited = self.p_no_decay * gaussian(I, self.mu_1, self.sigma_1)

        less_than = I < self.mu_0
        greater_than = I > self.mu_1

        t_s = self.tau_m * (I - self.mu_0) / (self.mu_1 - self.mu_0)
        p = self.p_decay * np.exp(-t_s / self.T1) / self.T1
        p[less_than] = 0
        p[greater_than] = 0

        b = gaussian(I, 0, self.sigma_1)
        decay = np.convolve(p, b, mode='same') / sum(b)

        return ground, excited, decay

    def sample(self, n):
        I = np.linspace(-2, 2, 10000)
        ground, excited, decay = self.probability(I)
        p = ground + excited + decay
        return np.random.choice(I, size = n, p = p / sum(p))

    def plot(self, I = None):
        if I is None:
            I = np.linspace(-2, 2, 10000)

        ground, excited, decay = self.probability(I)
        plt.plot(I, ground, label = 'ground', alpha = 0.5)
        plt.plot(I, excited + decay, label = 'decay', alpha = 0.5)
        plt.xlim(-1, 2)
        plt.legend()
        plt.show()


dist = Distribution()
dist.plot()

