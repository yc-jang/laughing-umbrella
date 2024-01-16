from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def groupby_pca_mean(df, group_columns, pca_components=2):
    """
    Perform PCA on the selected columns of a dataframe after grouping by specified columns.

    Parameters:
    - df: pandas DataFrame
    - group_columns: list of columns to group by
    - pca_components: number of components for PCA

    Returns:
    - DataFrame with the PCA results and mean values for each group
    """

    # Group by specified columns
    grouped = df.groupby(group_columns)

    # Initialize a list to store the results
    pca_results = []

    for name, group in grouped:
        # Select columns that are not in the group_by columns and are numerical
        non_group_columns = group.select_dtypes(include=[np.number]).drop(columns=group_columns)

        # Standardize the data before applying PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(non_group_columns)

        # Apply PCA
        pca = PCA(n_components=pca_components)
        pca_data = pca.fit_transform(scaled_data)

        # Compute the mean of the PCA components for the group
        mean_pca = np.mean(pca_data, axis=0)

        # Append the results
        pca_results.append([name, *mean_pca])

    # Create a DataFrame from the results
    columns = group_columns + [f'PCA_{i+1}' for i in range(pca_components)]
    pca_df = pd.DataFrame(pca_results, columns=columns)

    return pca_df

def groupby_pca_mean_corrected(df, group_columns, pca_components=2):
    """
    Perform PCA on the selected columns of a dataframe after grouping by specified columns.

    Parameters:
    - df: pandas DataFrame
    - group_columns: list of columns to group by
    - pca_components: number of components for PCA

    Returns:
    - DataFrame with the PCA results and mean values for each group
    """
    # Group by specified columns
    grouped = df.groupby(group_columns)

    # Initialize a list to store the results
    pca_results = []

    for name, group in grouped:
        # Select columns that are not in the group_by columns and are numerical
        non_group_columns = group.select_dtypes(include=[np.number]).drop(columns=group_columns)

        # Standardize the data before applying PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(non_group_columns)

        # Apply PCA
        pca = PCA(n_components=pca_components)
        pca_data = pca.fit_transform(scaled_data)

        # Compute the mean of the PCA components for the group
        mean_pca = np.mean(pca_data, axis=0)

        # Append the results (including the group name)
        pca_results.append([*name, *mean_pca])

    # Create a DataFrame from the results
    columns = group_columns + [f'PCA_{i+1}' for i in range(pca_components)]
    pca_df = pd.DataFrame(pca_results, columns=columns)

    return pca_df


def groupby_pca_dynamic_components(df, group_columns, corr_threshold=0.8, component_ratio=0.5):
    """
    Perform PCA on the selected columns of a dataframe after grouping by specified columns.
    PCA components are determined based on the number of highly correlated features.

    Parameters:
    - df: pandas DataFrame
    - group_columns: list of columns to group by
    - corr_threshold: threshold for considering high correlation
    - component_ratio: ratio to determine the number of PCA components based on correlated features

    Returns:
    - DataFrame with the PCA results for each group
    """
    # Group by specified columns
    grouped = df.groupby(group_columns)

    # Initialize a list to store the results
    pca_results = []

    for name, group in grouped:
        # Select columns that are not in the group_by columns and are numerical
        non_group_columns = group.select_dtypes(include=[np.number]).drop(columns=group_columns)

        # Calculate the correlation matrix and determine highly correlated pairs
        corr_matrix = non_group_columns.corr().abs()
        high_corr_vars = (corr_matrix > corr_threshold).sum() - 1  # Subtract 1 to ignore self-correlation
        high_corr_count = high_corr_vars[high_corr_vars > 0].count()

        # Determine the number of PCA components
        pca_components = max(int(high_corr_count * component_ratio), 1)

        # Standardize the data before applying PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(non_group_columns)

        # Apply PCA
        pca = PCA(n_components=pca_components)
        pca_data = pca.fit_transform(scaled_data)

        # Compute the mean of the PCA components for the group
        mean_pca = np.mean(pca_data, axis=0)

        # Append the results (including the group name)
        pca_result = [*name, *mean_pca]
        pca_results.append(pca_result)

    # Create a DataFrame from the results
    columns = group_columns + [f'PCA_{i+1}' for i in range(len(pca_results[0]) - len(group_columns))]
    pca_df = pd.DataFrame(pca_results, columns=columns)

    return pca_df


# Redefining the function with handling NaN values
def groupby_pca_dynamic_components(df, group_columns, corr_threshold=0.8, component_ratio=0.5):
    """
    Perform PCA on the selected columns of a dataframe after grouping by specified columns.
    PCA components are determined based on the number of highly correlated features.
    NaN values are handled by replacing them with zeros.

    Parameters:
    - df: pandas DataFrame
    - group_columns: list of columns to group by
    - corr_threshold: threshold for considering high correlation
    - component_ratio: ratio to determine the number of PCA components based on correlated features

    Returns:
    - DataFrame with the PCA results for each group
    """
    # Group by specified columns
    grouped = df.groupby(group_columns)

    # Initialize a list to store the results
    pca_results = []

    for name, group in grouped:
        # Select columns that are not in the group_by columns and are numerical
        non_group_columns = group.select_dtypes(include=[np.number]).drop(columns=group_columns)

        # Calculate the correlation matrix and determine highly correlated pairs
        corr_matrix = non_group_columns.corr().abs()
        high_corr_vars = (corr_matrix > corr_threshold).sum() - 1  # Subtract 1 to ignore self-correlation
        high_corr_count = high_corr_vars[high_corr_vars > 0].count()

        # Determine the number of PCA components
        pca_components = max(int(high_corr_count * component_ratio), 1)
        pca_components = min(pca_components, len(non_group_columns.columns))  # Ensure PCA components are not more than available features

        # Standardize the data before applying PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(non_group_columns)

        # Apply PCA
        pca = PCA(n_components=pca_components)
        pca_data = pca.fit_transform(scaled_data)

        # Compute the mean of the PCA components for the group
        mean_pca = np.nanmean(pca_data, axis=0)  # Handling NaN values by replacing them with the mean of the column

        # Append the results (including the group name)
        pca_result = [*name, *mean_pca]
        pca_results.append(pca_result)

    # Create a DataFrame from the results
    columns = group_columns + [f'PCA_{i+1}' for i in range(len(pca_results[0]) - len(group_columns))]
    pca_df = pd.DataFrame(pca_results, columns=columns)

    return pca_df
