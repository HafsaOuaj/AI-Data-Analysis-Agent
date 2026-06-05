import pandas as pd
from tools.data_understanding import load_dataset
import os
import sys

# Ensure the project source directory is on sys.path so local modules can be imported.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def correlation(path: str, col1: str, col2: str):
    """
    Compute correlation between two numeric columns.

    Args:
        path (str): Dataset path.
        col1 (str): First column.
        col2 (str): Second column.

    Returns:
        dict: Correlation value.
    """

    data = load_dataset(path)

    return {
        "correlation": float(data[col1].corr(data[col2]))
    }

def compare_groups(path: str, group_col: str, target_col: str):
    """
    Compare mean values across groups.

    Args:
        path (str): Dataset path.
        group_col (str): Categorical column.
        target_col (str): Numeric column.

    Returns:
        dict: Group-wise averages.
    """

    data = load_dataset(path)

    grouped = data.groupby(group_col)[target_col].mean()

    return grouped.to_dict()