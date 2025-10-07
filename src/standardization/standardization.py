import numpy as np


class Standardization:

    def __init__(self, entries: list[float]):
        self.entries = np.array(entries)
        self.mean = self._mean()
        self.variance = self._variance()
        self.standard_deviation = np.sqrt(self.variance)
        self.standardized_variable = self._standardized_variable()

    def _mean(self):
        return np.mean(self.entries)

    def _variance(self):
        return np.var(self.entries, ddof=0)

    def _standardized_variable(self):
        return (self.entries - self.mean) / self.standard_deviation

    def get_mean(self):
        return self.mean

    def get_variance(self):
        return self.variance

    def get_standard_deviation(self):
        return self.standard_deviation

    def get_standardized_variable(self):
        return self.standardized_variable