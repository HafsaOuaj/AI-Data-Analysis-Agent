import pandas as pd
from tools.data_understanding import load_dataset
import os
import sys

# Ensure the project source directory is on sys.path so local modules can be imported.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def get_total_revenue(path: str, revenue_column: str = "total_spent_usd"):
    """
    Compute total revenue from the dataset.

    This tool aggregates all customer spending into a single KPI.

    Args:
        path (str): Path to dataset file (CSV or Parquet).
        revenue_column (str): Column representing monetary value per customer/order.

    Returns:
        dict: Total revenue value.
    """

    data = load_dataset(path)

    if revenue_column not in data.columns:
        raise ValueError(f"Column '{revenue_column}' not found")

    total_revenue = data[revenue_column].sum()

    return {
        "total_revenue": float(total_revenue)
    }


def get_avg_order_value(path: str, revenue_column: str = "total_spent_usd"):
    """
    Compute average order value (AOV).

    AOV = total revenue / number of records (orders or customers depending on dataset).

    Args:
        path (str): Path to dataset file.
        revenue_column (str): Column representing monetary value.

    Returns:
        dict: Average order value.
    """

    data = load_dataset(path)

    if revenue_column not in data.columns:
        raise ValueError(f"Column '{revenue_column}' not found")

    aov = data[revenue_column].mean()

    return {
        "average_order_value": float(aov)
    }

def get_customer_lifetime_value(path: str, customer_column: str, revenue_column: str):
    """
    Compute average customer lifetime value (CLV).

    CLV is approximated as total revenue per customer.

    Args:
        path (str): Path to dataset file.
        customer_column (str): Customer identifier column.
        revenue_column (str): Revenue/spending column.

    Returns:
        dict: Average CLV and per-customer CLV distribution summary.
    """

    data = load_dataset(path)

    if customer_column not in data.columns:
        raise ValueError(f"Column '{customer_column}' not found")

    if revenue_column not in data.columns:
        raise ValueError(f"Column '{revenue_column}' not found")

    clv = (
        data.groupby(customer_column)[revenue_column]
        .sum()
    )

    return {
        "average_clv": float(clv.mean()),
        "min_clv": float(clv.min()),
        "max_clv": float(clv.max())
    }


def get_churn_rate(path: str, churn_column: str = "churn"):
    """
    Compute churn rate from dataset.

    Churn rate = percentage of customers marked as churned (1/True).

    Args:
        path (str): Path to dataset file.
        churn_column (str): Column indicating churn status (0/1 or True/False).

    Returns:
        dict: Churn rate percentage.
    """

    data = load_dataset(path)

    if churn_column not in data.columns:
        raise ValueError(f"Column '{churn_column}' not found")

    churn_rate = data[churn_column].mean() * 100

    return {
        "churn_rate_percent": float(churn_rate)
    }


def get_return_rate(path: str, return_column: str = "return_rate"):
    """
    Compute average return rate.

    Args:
        path (str): Path to dataset file.
        return_column (str): Column representing return percentage per customer/order.

    Returns:
        dict: Average return rate.
    """

    data = load_dataset(path)

    if return_column not in data.columns:
        raise ValueError(f"Column '{return_column}' not found")

    return {
        "average_return_rate": float(data[return_column].mean())
    }

def get_discount_usage_rate(path: str, discount_column: str = "uses_discount"):
    """
    Compute percentage of customers/orders using discounts.

    Args:
        path (str): Path to dataset file.
        discount_column (str): Binary column (0/1 or True/False).

    Returns:
        dict: Discount usage rate in percentage.
    """

    data = load_dataset(path)

    if discount_column not in data.columns:
        raise ValueError(f"Column '{discount_column}' not found")

    return {
        "discount_usage_rate_percent": float(data[discount_column].mean() * 100)
    }


def get_revenue_per_customer(path: str, customer_column: str, revenue_column: str):
    """
    Compute revenue generated per customer.

    Args:
        path (str): Path to dataset file.
        customer_column (str): Customer ID column.
        revenue_column (str): Revenue column.

    Returns:
        dict: Summary of revenue per customer distribution.
    """

    data = load_dataset(path)

    if customer_column not in data.columns:
        raise ValueError(f"Column '{customer_column}' not found")

    if revenue_column not in data.columns:
        raise ValueError(f"Column '{revenue_column}' not found")

    grouped = data.groupby(customer_column)[revenue_column].sum()

    return {
        "mean_revenue_per_customer": float(grouped.mean()),
        "median_revenue_per_customer": float(grouped.median()),
        "top_customer_revenue": float(grouped.max())
    }

if __name__ == "__main__":
    results =get_churn_rate("data/ecommerce_customer_analytics.csv")
    print(results)