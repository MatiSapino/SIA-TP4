import numpy as np

EUCLIDEAN = 'euclidean'
EXPONENTIAL = 'exponential'

class Kohonen:

    def __init__(self, data: np.ndarray, grid_size: int, neurons: int, epochs: int, radio: int, radio_constant: bool, similarity_metric: str):
        self.data = data
        self.num_records, self.input_dim = data.shape
        self.grid_size = grid_size
        self.neurons = neurons
        self.epochs = epochs
        self.radio = radio
        self.radio_constant = radio_constant
        self.learning_rate = 1
        self.similarity_metric = similarity_metric
        self.neuron_coords = self._get_neuron_coordinates()
        self.weights = self._initialize_weights()

    def _get_neuron_coordinates(self) -> np.ndarray:
        return np.array(np.mgrid[:self.grid_size, :self.grid_size].T.reshape(self.neurons, 2))

    def _initialize_weights(self) -> np.ndarray:
        random_indices = np.random.choice(
            self.num_records,
            size=self.neurons,
            replace=True
        )
        return self.data[random_indices]

    def _update_learning_params(self, epoch: int):
        i = epoch + 1
        if self.radio_constant == False and self.radio > 1:
            self.radio -= 1

        self.learning_rate = 1 / i

    def _get_bmu_index(self, pattern: np.ndarray) -> int:
        if self.similarity_metric == EUCLIDEAN:
            distances = np.linalg.norm(pattern - self.weights, axis=1)
            return int(np.argmin(distances))
        else:
            distances_sq = np.sum((pattern - self.weights) ** 2, axis=1)
            similitude = np.exp(-distances_sq)
            return int(np.argmax(similitude))

    def _get_neighborhood_influence(self, bmu_index: int) -> np.ndarray:
        bmu_coord = self.neuron_coords[bmu_index]
        grid_distances_sq = np.sum((self.neuron_coords - bmu_coord) ** 2, axis=1)
        r_sq = 2 * (self.radio ** 2)
        return np.exp(-grid_distances_sq / r_sq)

    def _update_weight(self, pattern: np.ndarray, bmu_index: int):
        neighborhood_influence = self._get_neighborhood_influence(bmu_index)
        learning_rate_factor = self.learning_rate * neighborhood_influence
        correction_term = pattern - self.weights
        update = learning_rate_factor[:, np.newaxis] * correction_term
        self.weights += update

    def train(self):
        print(f"--- Starting Training ---")
        print(f"Map: {self.grid_size} x {self.grid_size}. Epochs: {self.epochs}")
        print(f"Weights initialized with random examples of the dataset")

        for epoch in range(self.epochs):
            self._update_learning_params(epoch)

            np.random.shuffle(self.data)

            for pattern in self.data:
                bmu_index = self._get_bmu_index(pattern)
                self._update_weight(pattern, bmu_index)

            print(f"Epoch: {epoch}, Radio = {self.radio}, Current Learning Rate = {self.learning_rate:.4f}")

        print(f"--- Finished Training ---")

    def get_bmus_for_data(self, data: np.ndarray) -> np.ndarray:
        num_records = data.shape[0]
        bmu_assignments = np.zeros(num_records, dtype=int)
        for record_index, pattern in enumerate(data):
            bmu_assignments[record_index] = self._get_bmu_index(pattern)
        return bmu_assignments