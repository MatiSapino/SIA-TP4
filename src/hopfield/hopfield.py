import numpy as np


def _initialize_weights(data: np.ndarray) -> np.ndarray:
    w = data.T @ data #uso la formula de la diapositiva 29. Pongo primero la transpuesta pues lo que en la teorica es K aca es data.T
    np.fill_diagonal(w, 0)
    print(data)
    print(data.T)
    print(data.shape)
    return w / data.shape[1] # data.shape devuelve la tupla (P , N) donde N es el numero de neuronas y P el numero de patrones almacenados.


class Hopfield:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.weights = _initialize_weights(self.data)
        self.neurons = np.zeros(self.data.shape[1])

    #Dado un patron ζ nuevo, devuelve el patron ξ almacenado mas parecido
    def evaluate(self, pattern: np.ndarray) -> np.ndarray:
        self.neurons = pattern.copy()
        prev_state = np.zeros_like(self.neurons)
        prev_prev_state = np.zeros_like(self.neurons)
        iteration = 0
        while not np.array_equal(prev_state, self.neurons) and not np.array_equal(prev_prev_state, self.neurons):
            prev_prev_state = prev_state.copy()
            prev_state = self.neurons.copy()
            print(f"Iteration {iteration}: Neuron States: {self.neurons}")
            self.neurons = np.sign(self.weights @ self.neurons)
            for i in range(len(self.neurons)):
                if self.neurons[i] == 0:
                    self.neurons[i] = prev_state[i]
            iteration += 1

        return self.neurons

    def evaluate_multiple_patterns(self, patterns: np.ndarray) -> np.ndarray:
        results = []
        for pattern in patterns:
            result = self.evaluate(pattern)
            results.append(result)
        return np.array(results)

    def print_weights(self):
        print(self.weights)

