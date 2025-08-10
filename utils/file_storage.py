"""
File storage utilities for PTSD ML Platform.
Handles saving, listing, and deleting result files locally in 'web_results'.
"""

import os
from datetime import datetime
import pandas as pd
from typing import Union


def save_to_results(data: Union[pd.DataFrame, dict], result_type: str = "result") -> str:
    """
    Save results to timestamped CSV or JSON file inside 'web_results' directory.

    Args:
        data (pd.DataFrame or dict): Results to save.
        result_type (str): Label for file prefix.

    Returns:
        str: Path to saved file.
    """
    os.makedirs("web_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert lists to strings if dealing with DataFrame columns containing lists
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            # Convert list objects in columns to strings
            if data[col].apply(lambda x: isinstance(x, list)).any():
                data[col] = data[col].apply(str)
        filename = f"{result_type}_results_{timestamp}.csv"
        file_path = os.path.join("web_results", filename)
        data.to_csv(file_path, index=False)

    elif isinstance(data, dict):
        filename = f"{result_type}_results_{timestamp}.json"
        file_path = os.path.join("web_results", filename)
        import json
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        raise TypeError("Data must be a pandas DataFrame or a dict.")

    return file_path


def list_results() -> list:
    """
    List all result files in 'web_results', sorted newest first.

    Returns:
        list: Filenames.
    """
    if not os.path.exists("web_results"):
        return []
    files = [
        f for f in os.listdir("web_results")
        if f.lower().endswith((".csv", ".json"))
    ]
    files.sort(key=lambda f: os.path.getmtime(os.path.join("web_results", f)), reverse=True)
    return files


def delete_result(filename: str) -> bool:
    """
    Delete a result file from 'web_results'.

    Args:
        filename (str): Filename to delete.

    Returns:
        bool: True if deleted, False if not found.
    """
    path = os.path.join("web_results", filename)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
