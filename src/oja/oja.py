import numpy as np


class Oja:

    def __init__(self, data: np.ndarray, epochs:int, learning_rate: float):
        self.data = data
        self.num_records, self.input_dim = data.shape
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(self.input_dim)
        self.weights /= np.linalg.norm(self.weights)
        self.final_weights = None

    def train(self):
        weights = self.weights.copy()

        for epoch in range(self.epochs):
            eta = self.learning_rate
            np.random.shuffle(self.data)

            for x in self.data:
                y = np.dot(x, weights)
                delta_w = eta * y * x - eta * (y ** 2) * weights
                weights += delta_w

        self.final_weights = weights
        return self.final_weights

    def get_pc1_weights(self):
        return self.final_weights