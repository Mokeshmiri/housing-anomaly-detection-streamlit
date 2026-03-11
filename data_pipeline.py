import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_and_prepare_data(sample_size: int = 10000) -> pd.DataFrame:
    """
    load the california housing dataset and return a cleaned table.

    each row represents a small area in california with basic housing and
    demographic information. here we do light cleaning so that the model can
    focus on finding unusual areas.
    """
    raw_data = fetch_california_housing(as_frame=True)
    full_frame = raw_data.frame.copy()

    if sample_size is not None and sample_size < len(full_frame):
        full_frame = full_frame.sample(n=sample_size, random_state=42)

    full_frame = full_frame.dropna(axis=1, how="all")

    numeric_columns = full_frame.select_dtypes(include=["number"]).columns.tolist()

    if numeric_columns:
        full_frame[numeric_columns] = full_frame[numeric_columns].fillna(
            full_frame[numeric_columns].median(numeric_only=True)
        )

    return full_frame.reset_index(drop=True)

