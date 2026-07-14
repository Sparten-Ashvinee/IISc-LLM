"""Data loading module — fetches LinkedIn job postings from Hugging Face."""

import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def load_linkedin_data(dataset_name: str = "arshkon/linkedin-job-postings",
                       max_records: int | None = 50000,
                       cache_dir: str = "./cache") -> pd.DataFrame:
    """Load LinkedIn job postings dataset from Hugging Face.

    Returns a cleaned DataFrame ready for chunking.
    """
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train", cache_dir=cache_dir)

    df = ds.to_pandas()

    if max_records:
        df = df.head(max_records)

    # Keep relevant columns (adapt based on actual dataset schema)
    keep_cols = [
        "job_id", "title", "company_name", "description",
        "location", "formatted_experience_level",
        "skills_desc", "job_type",
    ]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    # Basic cleaning
    df = df.dropna(subset=["description"])
    df["description"] = df["description"].astype(str).str.strip()
    df = df[df["description"].str.len() > 50]  # drop near-empty postings
    df = df.drop_duplicates(subset=["description"]).reset_index(drop=True)

    print(f"Loaded {len(df):,} job postings after cleaning")
    return df


if __name__ == "__main__":
    df = load_linkedin_data()
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    print(f"Nulls:\n{df.isnull().sum()}")
