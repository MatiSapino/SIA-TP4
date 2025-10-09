import numpy as np

from src.kohonen.kohonen import Kohonen, EUCLIDEAN
from src.standardization.standardization import Standardization

MAP_K = 2
NEURONS = MAP_K * MAP_K
EPOCHS = 500 * NEURONS
INITIAL_ETA = 0.5

def main():
    raw_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    num_records, num_variables = raw_data.shape
    standardization_data = np.zeros_like(raw_data, dtype=float)
    for col_index in range(num_variables):
        variable_data = raw_data[:, col_index].tolist()
        std_processor = Standardization(entries=variable_data)
        standardization_data[:, col_index] = std_processor.get_standardized_variable()

    kohonen = Kohonen(
        data=standardization_data,
        map_k=MAP_K,
        epochs=EPOCHS,
        initial_eta=INITIAL_ETA,
        similarity_metric=EUCLIDEAN
    )

    kohonen.train()

if __name__ == "__main__":
    main()