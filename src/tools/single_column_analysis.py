# Ensure the project source directory is in sys.path
import os
import sys
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
from tools.data_understanding import load_dataset
from mcp.server.fastmcp import FastMCP

# -----------------------
# LOGGING SETUP
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("single_column_analysis")
mcp = FastMCP("single_column_analysis")
def main():
    logger.info("Starting MCP server: single_column_analysis")
    mcp.run(transport="stdio")

def value_counts(path: str, column: str):
    """
    Compute frequency counts of values in a categorical column.

    This tool is used to analyze categorical distributions such as:
    customer segments, product categories, regions, etc.

    Args:
        path (str): Path to the dataset file (CSV or Parquet).
        column (str): Name of the categorical column to analyze.

    Returns:
        dict: A dictionary mapping each unique value to its frequency.

    Raises:
        ValueError: If the column does not exist or is not categorical.
    """

    data = load_dataset(path)

    if column not in data.columns:
        raise ValueError(f"Column {column} does not exist in dataset")

    if data[column].dtype not in ["object", "category","str"]:
        raise ValueError(f"Column {column} is not categorical")

    return data[column].value_counts().to_dict()


def distribution(path: str, column: str):
    """
    Compute statistical distribution of a numeric column.

    This tool provides summary statistics useful for understanding:
    central tendency, spread, and distribution shape.

    Args:
        path (str): Path to the dataset file (CSV or Parquet).
        column (str): Name of the numeric column.

    Returns:
        dict: Statistical metrics including mean, std, min, max, median, and skew.

    Raises:
        ValueError: If column does not exist or is not numeric.
    """

    data = load_dataset(path)

    if column not in data.columns:
        raise ValueError(f"Column {column} does not exist in dataset")

    if not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Column {column} is not numeric")

    return {
        "mean": data[column].mean(),
        "std": data[column].std(),
        "min": data[column].min(),
        "max": data[column].max(),
        "median": data[column].median(),
        "skew": data[column].skew(),
        "q1": data[column].quantile(0.25),
        "q3": data[column].quantile(0.75),
    }


def outliers_iqr(path: str, column: str):
    """
    Detect outliers in a numeric column using the IQR method.

    This tool identifies extreme values that lie outside
    1.5 * IQR range and is useful for anomaly detection.

    Args:
        path (str): Path to the dataset file (CSV or Parquet).
        column (str): Name of the numeric column to analyze.

    Returns:
        dict: Contains:
            - count (int): Number of outliers
            - lower_bound (float): Lower IQR threshold
            - upper_bound (float): Upper IQR threshold
            - outliers (list): List of outlier records

    Raises:
        ValueError: If column does not exist or is not numeric.
    """

    data = load_dataset(path)

    if column not in data.columns:
        raise ValueError(f"Column {column} does not exist in dataset")

    if not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Column {column} is not numeric")

    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]

    return {
        "count": len(outliers),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outliers": outliers.to_dict(orient="records")
    }
if __name__ == "__main__":
    main()