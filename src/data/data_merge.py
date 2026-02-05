"""
Merge and deduplicate property listing data from monthly scrapes
"""

import pandas as pd
from pathlib import Path
from glob import glob


def merge_and_deduplicate():
    """Merge all CSV files from raw folder and remove duplicates"""

    # Setup paths
    script_dir = Path(__file__).parent
    input_path = script_dir / "data" / "raw"
    output_path = script_dir / "data" / "processed" / "car_merged_deduplicated.csv"

    # Find all CSV files
    csv_files = sorted(glob(str(input_path / "car_cleaned_*.csv")))

    if not csv_files:
        print(f"No CSV files found in {input_path}")
        return

    # Read and merge all files
    print(f"Found {len(csv_files)} files. Merging...")
    merged_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Check and remove duplicates
    duplicate_count = merged_df.duplicated(subset='id').sum()
    print(f"Total rows: {len(merged_df):,}")
    print(f"Duplicates found: {duplicate_count:,}")

    # Remove duplicates (keep last occurrence as it's most recent)
    deduplicated_df = merged_df.drop_duplicates(subset='id', keep='last')
    print(f"Rows after deduplication: {len(deduplicated_df):,}")

    # Save merged file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deduplicated_df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    merge_and_deduplicate()
