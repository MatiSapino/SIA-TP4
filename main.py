import argparse
import json
import numpy as np
import pandas as pd

from src.kohonen.kohonen import Kohonen, EUCLIDEAN, EXPONENTIAL
from src.standardization.standardization import Standardization

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unsupervised Learning.')
    parser.add_argument('--config-file', type=str, default="./configs/config.json", help='Path to the configuration JSON file.')
    parser_args = parser.parse_args()

    with open(parser_args.config_file, 'r') as file:
        config = json.load(file)

    csv_file = config['csv_file']
    epochs_factor = config['epochs_factor']
    radio = config['r']
    radio_constant = config['r_constant']
    grid_size = config['k']
    similarity_metric = config['similarity_metric']
    if similarity_metric != EUCLIDEAN and similarity_metric != EXPONENTIAL:
        raise ValueError('Only euclidean or exponential are supported.')

    neurons = grid_size * grid_size
    epochs = epochs_factor * neurons

    df = pd.read_csv(csv_file)
    countries = df['Country'].values
    raw_data = df.drop('Country', axis=1).values
    num_records, num_variables = raw_data.shape
    standardization_data = np.zeros_like(raw_data, dtype=float)
    for col_index in range(num_variables):
        variable_data = raw_data[:, col_index].tolist()
        std_processor = Standardization(entries=variable_data)
        standardization_data[:, col_index] = std_processor.get_standardized_variable()

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