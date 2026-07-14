"""EDA script for LinkedIn job postings — run this or use in a notebook."""

import sys
sys.path.insert(0, "..")  # when running from notebooks/

import pandas as pd
import matplotlib.pyplot as plt
from data.load_data import load_linkedin_data


def run_eda():
    df = load_linkedin_data(max_records=10000)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
    print(f"\nDuplicates: {df.duplicated(subset=['description']).sum()}")

    # Text length distribution
    df["desc_len"] = df["description"].str.len()
    print(f"\nDescription length stats:\n{df['desc_len'].describe()}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["desc_len"], bins=50, edgecolor="black")
    axes[0].set_title("Job Description Length Distribution")
    axes[0].set_xlabel("Characters")
    axes[0].set_ylabel("Count")

    # Top fields
    if "formatted_experience_level" in df.columns:
        exp_counts = df["formatted_experience_level"].value_counts().head(10)
        exp_counts.plot.barh(ax=axes[1])
        axes[1].set_title("Experience Level Distribution")
        axes[1].set_xlabel("Count")

    plt.tight_layout()
    plt.savefig("../artifacts/eda_overview.png", dpi=100, bbox_inches="tight")
    plt.show()

    # Top companies
    if "company_name" in df.columns:
        print(f"\nTop 15 companies:")
        print(df["company_name"].value_counts().head(15))

    # Sample records
    print(f"\n{'=' * 60}")
    print("SAMPLE RECORDS (first 3)")
    print("=" * 60)
    for i, row in df.head(3).iterrows():
        print(f"\n--- Record {i} ---")
        print(f"Title: {row.get('title', 'N/A')}")
        print(f"Company: {row.get('company_name', 'N/A')}")
        print(f"Location: {row.get('location', 'N/A')}")
        print(f"Description (first 300 chars): {str(row.get('description', ''))[:300]}")

    return df


if __name__ == "__main__":
    import os
    os.makedirs("../artifacts", exist_ok=True)
    df = run_eda()
