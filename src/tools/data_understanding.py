# Ensure the project source directory is in sys.path
import os
import sys
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
from mcp.server.fastmcp import FastMCP

# -----------------------
# LOGGING SETUP
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("data_understanding")
mcp = FastMCP("data_understanding")
def main():
    logger.info("Starting MCP server: data_understanding")
    mcp.run(transport="stdio")


@mcp.tool()
def load_dataset(path:str):
    """
    Load a dataset from a CSV or Parquet file.

    This function reads a dataset from disk and converts it into
    a pandas DataFrame for further analysis.

    Supported formats:
        - .csv
        - .parquet

    Args:
        path (str): Path to the dataset file.

    Returns:
        pandas.DataFrame: Loaded dataset as a DataFrame.

    Raises:
        ValueError: If the file format is not supported.
    """
    if path.endswith("csv"):
        data = pd.read_csv(path)
    elif path.endswith("parquet"):
        data = pd.read_parquet(path)

    else:
        raise ValueError("Unsupported file format")
    return data

@mcp.tool()
def get_schema(path:str):
    """
    Generate structural metadata of the dataset.

    This tool provides an overview of the dataset structure,
    including column names, data types, missing values,
    and dataset shape.

    Args:
        path (str): Path to the dataset file.
    """
    data = load_dataset(path)
    return {
        "columns": list(data.columns),
        "dtypes": data.dtypes.astype(str).to_dict(),
        "missing_values": data.isnull().sum().to_dict()
    }

@mcp.tool()
def get_summary_stats(path:str):
    """
        Compute descriptive statistics for all columns in the dataset.

        This includes numerical and categorical summary statistics
        such as mean, median, standard deviation, counts, and unique values.

        Args:
            path (str): Path to the dataset file.
        """
    data = load_dataset(path)
    return data.describe(include="all").to_dict()
@mcp.tool()
def get_missing_values(path:str):
    """
    Calculate missing value counts for each column.

    This tool helps identify data quality issues by reporting
    the number of null or missing entries per column.

    Args:
        path (str): Path to the dataset file.

    """
    data =load_dataset(path)

    return data.isnull().sum().to_dict(orient="records")


if __name__ == "__main__":
    main()