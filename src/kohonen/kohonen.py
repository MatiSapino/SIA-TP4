import numpy as np

EUCLIDEAN = 'euclidean'

class Kohonen:

    def __init__(self, data: np.ndarray, map_k: int, epochs: int, initial_eta: float, eta_power_decay: float = 1.0, similarity_metric: str = EUCLIDEAN):
        self.data = data
        self.num_records, self.input_dim = data.shape
        self.map_k = map_k
        self.num_neurons = map_k * map_k
        self.initial_R = map_k
        self.current_R = map_k
        self.eta_power_decay = eta_power_decay
        self.initial_eta = initial_eta
        self.current_eta = initial_eta
        self.epochs = epochs
        self.similarity_metric = similarity_metric
        self.neuron_coords = self._get_neuron_coordinates()
        self.weights = self._initialize_weights()

    def _get_neuron_coordinates(self) -> np.ndarray:
        return np.array(np.mgrid[:self.map_k, :self.map_k].T.reshape(self.num_neurons, 2))

    def _initialize_weights(self) -> np.ndarray:
        random_indices = np.random.choice(
            self.num_records,
            size=self.num_neurons,
            replace=True
        )
        return self.data[random_indices]

    def _update_learning_params(self, epoch: int):
        i = epoch + 1
        if self.current_R > 1:
            self.current_R -= 1

        ## n(i) = n_o / i^p
        self.current_eta = self.initial_eta / (i ** self.eta_power_decay)

    def _get_bmu_index(self, pattern: np.ndarray) -> int:
        if self.similarity_metric == EUCLIDEAN:
            distances = np.linalg.norm(pattern - self.weights, axis=1)
            return np.argmin(distances)
        else:
            distances_sq = np.sum((pattern - self.weights) ** 2, axis=1)
            similitude = np.exp(-distances_sq)
            return np.argmax(similitude)

    def _get_neighborhood_influence(self, bmu_index: int) -> np.ndarray:
        bmu_coord = self.neuron_coords[bmu_index]
        grid_distances_sq = np.sum((self.neuron_coords - bmu_coord) ** 2, axis=1)
        r_sq = self.current_R ** 2
        is_inside_neighborhood = grid_distances_sq < r_sq
        return is_inside_neighborhood.astype(float)

    def _update_weight(self, pattern: np.ndarray, bmu_index: int):
        neighborhood_influence = self._get_neighborhood_influence(bmu_index)
        learning_rate_factor = self.current_eta * neighborhood_influence
        correction_term = pattern - self.weights
        update = learning_rate_factor[:, np.newaxis] * correction_term
        self.weights += update

    def train(self):
        print(f"--- Starting Training ---")
        print(f"Map: {self.map_k} x {self.map_k}. Epochs: {self.epochs}")
        print(f"Weights initialized with random examples of the dataset")

        for epoch in range(self.epochs):
            self._update_learning_params(epoch)

            np.random.shuffle(self.data)

            for pattern in self.data:
                bmu_index = self._get_bmu_index(pattern)
                self._update_weight(pattern, bmu_index)

            print(f"Epoch: {epoch} / {self.epochs}: R = {self.current_R}, eta = {self.current_eta:.4f}")

        print(f"--- Finished Training ---")