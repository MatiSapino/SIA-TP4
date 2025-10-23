import numpy as np


def _initialize_weights(data: np.ndarray) -> np.ndarray:
    w = data.T @ data #uso la formula de la diapositiva 29. Pongo primero la transpuesta pues lo que en la teorica es K aca es data.T
    np.fill_diagonal(w, 0)
    return w / data.shape[1] # data.shape devuelve la tupla (P , N) donde N es el numero de neuronas y P el numero de patrones almacenados.


class Hopfield:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.weights = _initialize_weights(self.data)
        self.neurons = np.zeros(self.data.shape[1])

    #Dado un patron ζ nuevo, devuelve el patron ξ almacenado mas parecido
    def train(self, pattern: np.ndarray) -> np.ndarray:
        self.neurons = pattern.copy()
        prev_state = np.zeros_like(self.neurons)
        iteration = 0
        while not np.array_equal(prev_state, self.neurons):
            prev_state = self.neurons.copy()
            print(f"Iteration {iteration}: Neuron States: {self.neurons}")
            self.neurons = np.sign(self.weights @ self.neurons)
            iteration += 1

        return self.neurons

    def print_weights(self):
        print(self.weights)

