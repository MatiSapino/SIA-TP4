import numpy as np


class Standardization:

    def __init__(self, entries: list[float]):
        self.entries = np.array(entries, dtype=float)
        self.mean = self._mean()
        self.variance = self._variance()
        self.standard_deviation = np.sqrt(self.variance)
        self.standardized_variable = self._standardized_variable()
        self.unit_lenght_scaling = self._unit_lenght_scaling()

    def _mean(self):
        return np.mean(self.entries)

    def _variance(self):
        return np.var(self.entries, ddof=0)

    def _standardized_variable(self):
        return (self.entries - self.mean) / self.standard_deviation

    def _unit_lenght_scaling(self):
        norm_2 = np.linalg.norm(self.entries)
        if norm_2 == 0:
            return np.zeros_like(self.entries)
        return self.entries / norm_2

    def get_mean(self):
        return self.mean

    def get_standard_deviation(self):
        return self.standard_deviation

    def get_standardized_variable(self):
        return self.standardized_variable

    def get_unit_lenght_scaling(self):
        return self.unit_lenght_scaling