from pathlib import Path
from typing import Tuple

import click
import pandas as pd


def get_dataset(
    csv_path: Path, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    target = "Cover_Type"
    data_features = dataset.drop(target, axis=1)
    data_target = dataset[target]
    return data_features, data_target