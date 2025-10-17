import argparse
import json
import numpy as np
import pandas as pd

from src.kohonen.kohonen import Kohonen, EUCLIDEAN, EXPONENTIAL
from src.oja.oja import Oja
from src.standardization.standardization import Standardization
from src.pca.pca import Pca
from sklearn.decomposition import PCA


def run_kohonen(config, standardization_data, countries):
    epochs_factor = config['epochs_factor']
    radio = config['r']
    radio_constant = config['r_constant']
    grid_size = config['k']
    similarity_metric = config['similarity_metric']
    if similarity_metric != EUCLIDEAN and similarity_metric != EXPONENTIAL:
        raise ValueError('Only euclidean or exponential are supported.')

    neurons = grid_size * grid_size
    epochs = epochs_factor * neurons

    kohonen = Kohonen(
        data=standardization_data,
        grid_size=grid_size,
        neurons=neurons,
        epochs=epochs,
        radio=radio,
        radio_constant=radio_constant,
        similarity_metric=similarity_metric
    )
    kohonen.train()

    print("\n--- Analyzing Country-to-Neuron Assignment ---")
    neuron_assignments = kohonen.get_bmus_for_data(standardization_data)
    neuron_counts = np.bincount(neuron_assignments, minlength=neurons)
    print(f"Total patterns (countries): {len(countries)}")
    print("Count of countries per neuron (index 0 to {}):".format(neurons - 1))
    print(neuron_counts)
    country_map = {}
    for country, bmu_index in zip(countries, neuron_assignments):
        if bmu_index not in country_map:
            country_map[bmu_index] = []
        country_map[bmu_index].append(country)

    print("\nAssociations of Countries to Neurons (BMU):")
    for bmu_index, country_list in country_map.items():
        print(f"Neuron {bmu_index}: {', '.join(country_list)}")

def run_oja(config, standardization_data, num_variables):
    learning_rate = config['n']
    epochs = config['epochs']

    print("--- PCA (Manual) ---")
    pca_manual = Pca(data=standardization_data)
    print("\nEigenvalues:")
    print(pca_manual.get_eigenvalues())
    print("\nFirst Principal Component PC1:")
    print(pca_manual.get_eigenvectors()[:, 0])
    print("\nFirst 5 Principal Component Rows (PC Scores - Y):")
    print(pca_manual.get_principal_components()[:5, :])
    print("\n")
    print("-" * 50)

    print("\n--- PCA (Sklearn) ---")
    pca_sklearn = PCA(n_components=num_variables)
    pca_scores_sklearn = pca_sklearn.fit_transform(standardization_data)
    print("\nEigenvalues:")
    print(pca_sklearn.explained_variance_)
    print("\nFirst Principal Component PC1:")
    print(pca_sklearn.components_[0])
    print("\nFirst 5 Principal Component Rows (PC Scores - Y):")
    print(pca_scores_sklearn[:5, :])
    print("\n")
    print("-" * 50)

    print("\n--- OJA ---")
    oja = Oja(
        data=standardization_data,
        epochs=epochs,
        learning_rate=learning_rate
    )
    oja_weights = oja.train()
    if np.dot(oja_weights, pca_sklearn.components_[0]) < 0:
        oja_weights_aligned = -oja_weights
        sign_note = " (Inverted Sign)"
    else:
        oja_weights_aligned = oja_weights
        sign_note = ""
    print("\nFirst Principal Component PC1" + sign_note + ":")
    print(oja_weights_aligned)
    print("\nFirst 5 Principal Component Rows (PC Scores - Y):")
    oja_pc_scores = standardization_data @ oja_weights_aligned
    print(oja_pc_scores[:5])
    oja_error = np.linalg.norm(oja_weights_aligned - pca_sklearn.components_[0]) ** 2
    print(f"\nSquared Error between Oja and Sklearn (||Oja - Sklearn||^2): {oja_error:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unsupervised Learning.')
    parser.add_argument('--config-file', type=str, default="./configs/config.json", help='Path to the configuration JSON file.')
    parser_args = parser.parse_args()

    with open(parser_args.config_file, 'r') as file:
        config_data = json.load(file)

    csv_file = config_data['csv_file']
    df = pd.read_csv(csv_file)
    countries_data = df['Country'].values
    raw_data = df.drop('Country', axis=1).values
    num_records_data, num_variables_data = raw_data.shape

    standardization_data_input = np.zeros_like(raw_data, dtype=float)
    for col_index in range(num_variables_data):
        variable_data = raw_data[:, col_index].tolist()
        std_processor = Standardization(entries=variable_data)
        standardization_data_input[:, col_index] = std_processor.get_standardized_variable()

    algorithm = config_data["algorithm"]
    if algorithm == "kohonen":
        run_kohonen(config_data, standardization_data_input, countries_data)
    elif algorithm == "oja":
        run_oja(config_data, standardization_data_input, num_variables_data)
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Use 'kohonen' or 'oja'.")