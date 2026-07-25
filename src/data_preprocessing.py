"""
Data Preprocessing Module for AI Weather Intelligence Platform

This module provides reusable functions for loading, inspecting, and performing
initial data cleaning on the weatherAUS dataset. Functions are designed to be
industry-standard with comprehensive documentation and error handling.

Author: Data Engineering Team
Date: 2026
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the weather dataset from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing the weather data
        
    Returns
    -------
    pd.DataFrame
        Loaded dataset
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist
    pd.errors.EmptyDataError
        If the CSV file is empty
        
    Examples
    --------
    >>> df = load_dataset('data/raw/weatherAUS.csv')
    >>> print(df.shape)
    """
    try:
        logger.info(f"Loading dataset from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded dataset with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get basic information about the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze
        
    Returns
    -------
    dict
        Dictionary containing:
        - shape: Tuple of (rows, columns)
        - dtypes: Dictionary of column types
        - size_mb: Dataset size in megabytes
        - columns: List of column names
        
    Examples
    --------
    >>> info = get_dataset_info(df)
    >>> print(info['shape'])
    (150000, 23)
    """
    info = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'size_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        'columns': df.columns.tolist(),
        'index_name': df.index.name,
        'total_cells': df.shape[0] * df.shape[1]
    }
    logger.info(f"Dataset shape: {info['shape']}, Size: {info['size_mb']} MB")
    return info


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - Column: Column name
        - Missing_Count: Number of missing values
        - Missing_Percentage: Percentage of missing values
        - Data_Type: Data type of column
        
    Examples
    --------
    >>> missing = check_missing_values(df)
    >>> print(missing[missing['Missing_Percentage'] > 0])
    """
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': round(100 * df.isnull().sum().values / len(df), 2),
        'Data_Type': df.dtypes.values
    })
    
    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    ).reset_index(drop=True)
    
    logger.info(f"Found {len(missing_data)} columns with missing values")
    return missing_data


def check_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check for duplicate rows in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze
        
    Returns
    -------
    dict
        Dictionary containing:
        - total_duplicates: Total number of duplicate rows
        - duplicate_percentage: Percentage of duplicates
        - duplicates_df: DataFrame of duplicate rows
        
    Examples
    --------
    >>> dup_info = check_duplicates(df)
    >>> print(f"Duplicates: {dup_info['total_duplicates']}")
    """
    total_duplicates = df.duplicated().sum()
    duplicate_percentage = round(100 * total_duplicates / len(df), 2)
    
    duplicates_df = df[df.duplicated(keep=False)].sort_values(
        by=list(df.columns)
    ) if total_duplicates > 0 else pd.DataFrame()
    
    logger.info(f"Found {total_duplicates} duplicate rows ({duplicate_percentage}%)")
    
    return {
        'total_duplicates': total_duplicates,
        'duplicate_percentage': duplicate_percentage,
        'duplicates_df': duplicates_df
    }


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get comprehensive summary statistics for numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze
        
    Returns
    -------
    pd.DataFrame
        Summary statistics including mean, median, std, min, max, quartiles
        
    Examples
    --------
    >>> stats = get_summary_statistics(df)
    >>> print(stats)
    """
    numeric_df = df.select_dtypes(include=[np.number])
    logger.info(f"Summary statistics calculated for {len(numeric_df.columns)} numeric columns")
    return numeric_df.describe().T


def analyze_target_variable(df: pd.DataFrame, target_col: str = 'RainTomorrow') -> Dict[str, Any]:
    """
    Analyze the target variable for classification tasks.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze
    target_col : str
        Name of the target column (default: 'RainTomorrow')
        
    Returns
    -------
    dict
        Dictionary containing:
        - value_counts: Distribution of target variable
        - percentages: Percentage distribution
        - imbalance_ratio: Ratio of majority to minority class
        
    Examples
    --------
    >>> target_info = analyze_target_variable(df)
    >>> print(target_info['imbalance_ratio'])
    """
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found in dataset")
        return {}
    
    value_counts = df[target_col].value_counts()
    percentages = round(100 * df[target_col].value_counts(normalize=True), 2)
    
    if len(value_counts) == 2:
        imbalance_ratio = max(value_counts) / min(value_counts)
    else:
        imbalance_ratio = None
    
    logger.info(f"Target variable analysis: {dict(value_counts)}")
    
    return {
        'value_counts': value_counts,
        'percentages': percentages,
        'imbalance_ratio': imbalance_ratio,
        'missing_values': df[target_col].isnull().sum()
    }


def get_column_types(df: pd.DataFrame) -> Dict[str, list]:
    """
    Categorize columns by their data types.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze
        
    Returns
    -------
    dict
        Dictionary with categories:
        - numeric: Numeric columns
        - categorical: Categorical columns
        - datetime: Datetime columns
        - object: Object columns
        
    Examples
    --------
    >>> col_types = get_column_types(df)
    >>> print(col_types['numeric'])
    """
    column_types = {
        'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime': df.select_dtypes(include=['datetime64']).columns.tolist(),
    }
    
    logger.info(f"Column categorization: {len(column_types['numeric'])} numeric, "
               f"{len(column_types['categorical'])} categorical, "
               f"{len(column_types['datetime'])} datetime")
    
    return column_types


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive data quality checks.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze
        
    Returns
    -------
    dict
        Dictionary containing various data quality metrics
        
    Examples
    --------
    >>> quality = check_data_quality(df)
    >>> print(quality['completeness_score'])
    """
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    completeness_score = round(100 * (1 - missing_cells / total_cells), 2)
    uniqueness_score = round(100 * (1 - duplicate_rows / len(df)), 2)
    
    quality_report = {
        'total_rows': df.shape[0],
        'total_columns': df.shape[1],
        'total_cells': total_cells,
        'missing_cells': missing_cells,
        'duplicate_rows': duplicate_rows,
        'completeness_score': completeness_score,
        'uniqueness_score': uniqueness_score,
        'quality_score': round((completeness_score + uniqueness_score) / 2, 2)
    }
    
    logger.info(f"Data quality score: {quality_report['quality_score']}%")
    
    return quality_report


def print_dataset_overview(df: pd.DataFrame, target_col: str = 'RainTomorrow') -> None:
    """
    Print a comprehensive overview of the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze
    target_col : str
        Name of the target column (default: 'RainTomorrow')
        
    Examples
    --------
    >>> print_dataset_overview(df)
    """
    print("\n" + "="*80)
    print("DATASET OVERVIEW".center(80))
    print("="*80 + "\n")
    
    # Basic info
    info = get_dataset_info(df)
    print(f"Shape: {info['shape'][0]:,} rows × {info['shape'][1]:,} columns")
    print(f"Memory Usage: {info['size_mb']} MB\n")
    
    # Data Quality
    quality = check_data_quality(df)
    print(f"Data Quality Score: {quality['quality_score']}%")
    print(f"  - Completeness: {quality['completeness_score']}%")
    print(f"  - Uniqueness: {quality['uniqueness_score']}%\n")
    
    # Column Types
    col_types = get_column_types(df)
    print(f"Column Types:")
    print(f"  - Numeric: {len(col_types['numeric'])}")
    print(f"  - Categorical: {len(col_types['categorical'])}")
    print(f"  - Datetime: {len(col_types['datetime'])}\n")
    
    # Missing Values
    missing = check_missing_values(df)
    if len(missing) > 0:
        print(f"Columns with Missing Values: {len(missing)}")
        print(missing.head().to_string(index=False))
    else:
        print("No missing values found!\n")
    
    # Duplicates
    dup_info = check_duplicates(df)
    print(f"\nDuplicate Rows: {dup_info['total_duplicates']} ({dup_info['duplicate_percentage']}%)")
    
    # Target Variable
    if target_col in df.columns:
        target_info = analyze_target_variable(df, target_col)
        print(f"\nTarget Variable ('{target_col}'):")
        print(target_info['value_counts'].to_string())
        print(f"Distribution: {target_info['percentages'].to_string()}")
    
    print("\n" + "="*80 + "\n")
