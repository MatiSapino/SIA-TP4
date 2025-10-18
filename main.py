import argparse
import json
import numpy as np
import pandas as pd
import os
import json as _json
import matplotlib.pyplot as plt

from matplotlib import cm
from src.kohonen.kohonen import Kohonen, EUCLIDEAN, EXPONENTIAL
from src.oja.oja import Oja
from src.standardization.standardization import Standardization
from src.pca.pca import Pca
from sklearn.decomposition import PCA

def _create_biplot(components, pc_scores, countries, output_dir, var_names=None):
    """Create and save a biplot (PC2 vs PC1) with variable vectors and country labels."""
    pc1 = pc_scores[:, 0]
    pc2 = pc_scores[:, 1]
    fig, ax = plt.subplots(figsize=(10, 8))
    # scatter countries
    ax.scatter(pc1, pc2, c='tab:blue', alpha=0.7)
    for i, txt in enumerate(countries):
        ax.annotate(txt, (pc1[i], pc2[i]), fontsize=8)

    # plot variable vectors (loadings)
    loadings = components.T[:, :2]  # shape (n_vars, 2)
    # scale vectors for visibility: multiply by a constant
    scalex = 1.0
    scaley = 1.0
    for i, (x, y) in enumerate(loadings):
        ax.arrow(0, 0, x * scalex, y * scaley, color='r', alpha=0.8, head_width=0.03)
        name = var_names[i] if var_names is not None else f'Var{i}'
        ax.text(x * scalex * 1.05, y * scaley * 1.05, name, color='r', fontsize=9)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA Biplot (PC2 vs PC1)')
    ax.grid(True, linestyle='--', alpha=0.4)
    biplot_path = os.path.join(output_dir, 'pca_biplot.png')
    fig.tight_layout()
    fig.savefig(biplot_path, dpi=200)
    plt.close(fig)
    return biplot_path


def _create_index_plot(pc_scores, countries, output_dir):
    """Create and save an index plot (bar chart) for PC1 by country. Keep original order."""
    pc1 = pc_scores[:, 0]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(countries))
    ax.bar(x, pc1, color=cm.tab20.colors[:len(countries)])
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=90)
    ax.set_ylabel('PC1')
    ax.set_title('PC1 index plot by country')
    fig.tight_layout()
    index_path = os.path.join(output_dir, 'pca_index_plot.png')
    fig.savefig(index_path, dpi=200)
    plt.close(fig)
    return index_path


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


def run_pca_manual(standardization_data):
    print("--- PCA (Manual) ---")
    pca_manual = Pca(data=standardization_data)
    eigenvalues = pca_manual.get_eigenvalues()
    eigenvectors = pca_manual.get_eigenvectors()
    pc_scores = pca_manual.get_principal_components()
    print("\nEigenvalues:")
    print(eigenvalues)
    print("\nFirst Principal Component PC1:")
    print(eigenvectors[:, 0])
    print("\nFirst 5 Principal Component Rows (PC Scores - Y):")
    print(pc_scores[:5, :])
    print("\n")
    print("-" * 50)


def run_pca_sklearn(standardization_data, num_variables, countries=None, var_names=None):
    print("--- PCA (Sklearn) ---")
    pca_sklearn = PCA(n_components=num_variables)
    pca_scores_sklearn = pca_sklearn.fit_transform(standardization_data)
    eigenvalues = pca_sklearn.explained_variance_
    components = pca_sklearn.components_
    explained_ratio = pca_sklearn.explained_variance_ratio_
    print("\nEigenvalues:")
    print(eigenvalues)
    print("\nFirst Principal Component PC1:")
    print(components[0])
    print("\n")
    print("-" * 50)

    # Prepare outputs directory for PCA
    output_dir = os.path.join(os.getcwd(), 'outputs', 'pca_sklearn')
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata (eigenvalues, components, explained_ratio)
    meta = {
        'eigenvalues': [float(np.round(x, 6)) for x in eigenvalues.tolist()],
        'explained_variance_ratio': [float(np.round(x, 6)) for x in explained_ratio.tolist()],
        'components': [[float(np.round(v, 6)) for v in row.tolist()] for row in components]
    }
    meta_path = os.path.join(output_dir, 'pca_sklearn_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as mf:
        _json.dump(meta, mf, indent=2)

    # Save PC scores (unlabeled)
    pc_scores_path = os.path.join(output_dir, 'pca_sklearn_scores.csv')
    pd.DataFrame(pca_scores_sklearn).to_csv(pc_scores_path, index=False)

    # Save labeled CSV and create plots
    try:
        countries_list = list(countries)
    except Exception:
        countries_list = [str(c) for c in countries]
    df_scores = pd.DataFrame(pca_scores_sklearn, columns=[f'PC{i+1}' for i in range(pca_scores_sklearn.shape[1])])
    df_scores.insert(0, 'Country', countries_list)
    labeled_scores_path = os.path.join(output_dir, 'pca_sklearn_scores_labeled.csv')
    df_scores.to_csv(labeled_scores_path, index=False)

    # create plots
    _create_biplot(components, pca_scores_sklearn, countries_list, output_dir, var_names=var_names)
    _create_index_plot(pca_scores_sklearn, countries_list, output_dir)

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
    parser.add_argument('--config-file', type=str, default="./config/kohonen/config.json", help='Path to the configuration JSON file.')
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
    elif algorithm == "pca_manual":
        run_pca_manual(standardization_data_input)
    elif algorithm == "pca_sklearn":
        var_names = df.drop('Country', axis=1).columns.tolist()
        run_pca_sklearn(standardization_data_input, num_variables_data, countries=countries_data, var_names=var_names)
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Use 'kohonen', 'oja', 'pca_manual', or 'pca_sklearn'.")