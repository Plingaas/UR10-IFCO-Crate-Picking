import numpy as np


class LowPassFilter:

    def __init__(self, sample_freq, cutoff_freq, values=1) -> None:
        self.f_s = sample_freq
        self.f_c = cutoff_freq
        self.alpha = self.compute_alpha()
        self.y = np.zeros((values, 1))

    def compute_alpha(self):
        dt = 1.0 / self.f_s
        rc = 1.0 / (2 * np.pi * self.f_c)
        return dt / (rc + dt)

    def filter(self, z):
        if len(z) != len(self.y):
            raise Exception

        for i in range(len(self.y)):
            self.y[i] = self.alpha * z[i] + (1 - self.alpha) * self.y[i]

        return self.y