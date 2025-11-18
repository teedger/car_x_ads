"""
Multi-file data loader for car ads CSV files
Loads and combines 12 months of car data with temporal tracking
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from datetime import datetime
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_FILE_PATTERN,
    COLUMN_DTYPES, DATE_COLUMNS, CHUNK_SIZE, LOW_MEMORY
)
from src.utils.logger import get_logger, log_execution_time, log_dataframe_info
from src.utils.checkpoint_manager import DataFrameCheckpointManager


logger = get_logger(__name__)


class CarDataLoader:
    """
    Loads car advertisement data from multiple CSV files
    Handles 12 months of data with temporal tracking
    """

    def __init__(self, data_dir=None, file_pattern=None):
        """
        Initialize data loader

        Args:
            data_dir: Directory containing CSV files
            file_pattern: Pattern to match CSV files (e.g., 'car_cleaned_*.csv')
        """
        self.data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
        self.file_pattern = file_pattern if file_pattern else DATA_FILE_PATTERN
        self.logger = logger

    def get_data_files(self):
        """
        Get list of all data files matching the pattern

        Returns:
            Sorted list of file paths
        """
        pattern = str(self.data_dir / self.file_pattern)
        files = sorted(glob(pattern))

        self.logger.info(f"Found {len(files)} data files")

        if len(files) == 0:
            self.logger.warning(f"No files found matching: {pattern}")

        return files

    def extract_date_from_filename(self, filepath):
        """
        Extract collection date from filename

        Filename format: car_cleaned_YYYYMMDD.csv

        Args:
            filepath: Path to CSV file

        Returns:
            datetime object or None if parsing fails
        """
        try:
            filename = Path(filepath).stem  # Get filename without extension
            # Extract date part (last 8 characters should be YYYYMMDD)
            date_str = filename.split('_')[-1]  # Get last part after underscore

            # Parse date
            if len(date_str) == 8 and date_str.isdigit():
                date = datetime.strptime(date_str, '%Y%m%d')
                return date
            else:
                self.logger.warning(f"Could not parse date from: {filename}")
                return None

        except Exception as e:
            self.logger.error(f"Error extracting date from {filepath}: {e}")
            return None

    def load_single_file(self, filepath, add_collection_date=True):
        """
        Load a single CSV file

        Args:
            filepath: Path to CSV file
            add_collection_date: Add collection_date column

        Returns:
            pandas DataFrame
        """
        self.logger.info(f"Loading: {Path(filepath).name}")

        try:
            # Load CSV
            df = pd.read_csv(
                filepath,
                low_memory=LOW_MEMORY,
                parse_dates=DATE_COLUMNS
            )

            # Add collection date
            if add_collection_date:
                collection_date = self.extract_date_from_filename(filepath)
                if collection_date:
                    df['collection_date'] = collection_date
                else:
                    df['collection_date'] = pd.NaT

            self.logger.info(f"  Loaded {len(df):,} rows")

            return df

        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {e}")
            return None

    @log_execution_time
    def load_all_files(self, use_checkpoint=True, checkpoint_name="merged_data"):
        """
        Load and combine all CSV files

        Args:
            use_checkpoint: Use checkpoint to resume if interrupted
            checkpoint_name: Name for checkpoint file

        Returns:
            Combined pandas DataFrame
        """
        # Check for existing checkpoint
        if use_checkpoint:
            checkpoint_mgr = DataFrameCheckpointManager(
                PROCESSED_DATA_DIR,
                checkpoint_name,
                format='parquet'
            )

            if checkpoint_mgr.exists():
                self.logger.info("Loading from checkpoint...")
                df, metadata = checkpoint_mgr.load()
                if df is not None:
                    return df

        # Get list of files
        files = self.get_data_files()

        if len(files) == 0:
            self.logger.error("No data files found!")
            return None

        # Load all files with progress bar
        dfs = []
        self.logger.info(f"Loading {len(files)} files...")

        for filepath in tqdm(files, desc="Loading files"):
            df = self.load_single_file(filepath)
            if df is not None:
                dfs.append(df)

        # Combine all DataFrames
        self.logger.info("Combining all data...")
        combined_df = pd.concat(dfs, ignore_index=True)

        self.logger.info(f"Combined data shape: {combined_df.shape}")
        log_dataframe_info(combined_df, "Combined Data", self.logger)

        # Save checkpoint
        if use_checkpoint:
            self.logger.info("Saving checkpoint...")
            checkpoint_mgr.save(
                combined_df,
                metadata={'n_files': len(files), 'stage': 'merged'}
            )

        return combined_df

    def load_files_chunked(self, chunk_size=None):
        """
        Load files in chunks to save memory
        Yields chunks of data instead of loading all at once

        Args:
            chunk_size: Number of rows per chunk

        Yields:
            DataFrame chunks
        """
        chunk_size = chunk_size if chunk_size else CHUNK_SIZE
        files = self.get_data_files()

        self.logger.info(f"Loading {len(files)} files in chunks of {chunk_size:,}")

        for filepath in tqdm(files, desc="Processing files"):
            try:
                # Read file in chunks
                for chunk in pd.read_csv(
                    filepath,
                    chunksize=chunk_size,
                    low_memory=LOW_MEMORY,
                    parse_dates=DATE_COLUMNS
                ):
                    # Add collection date
                    collection_date = self.extract_date_from_filename(filepath)
                    if collection_date:
                        chunk['collection_date'] = collection_date

                    yield chunk

            except Exception as e:
                self.logger.error(f"Error reading {filepath}: {e}")

    def get_file_summary(self):
        """
        Get summary information about all data files

        Returns:
            DataFrame with file summary info
        """
        files = self.get_data_files()

        summaries = []
        for filepath in tqdm(files, desc="Analyzing files"):
            try:
                # Get basic file info
                file_path = Path(filepath)
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB

                # Load just to get row count and columns
                df = pd.read_csv(filepath, nrows=1)
                row_count = sum(1 for _ in open(filepath)) - 1  # Subtract header

                collection_date = self.extract_date_from_filename(filepath)

                summaries.append({
                    'filename': file_path.name,
                    'collection_date': collection_date,
                    'file_size_mb': round(file_size, 2),
                    'row_count': row_count,
                    'n_columns': len(df.columns)
                })

            except Exception as e:
                self.logger.error(f"Error analyzing {filepath}: {e}")

        summary_df = pd.DataFrame(summaries)

        if len(summary_df) > 0:
            summary_df = summary_df.sort_values('collection_date')

        return summary_df


# Helper functions
def quick_load():
    """
    Quick load function for interactive use

    Returns:
        Combined DataFrame
    """
    loader = CarDataLoader()
    return loader.load_all_files()


def get_summary():
    """
    Get summary of all data files

    Returns:
        Summary DataFrame
    """
    loader = CarDataLoader()
    return loader.get_file_summary()


# Example usage
if __name__ == "__main__":
    from src.utils.logger import setup_logger

    # Setup logging
    setup_logger("car_data_loader", level='INFO')

    # Create loader
    loader = CarDataLoader()

    # Get file summary
    print("\n" + "=" * 70)
    print("FILE SUMMARY")
    print("=" * 70)
    summary = loader.get_file_summary()
    print(summary.to_string(index=False))
    print(f"\nTotal files: {len(summary)}")
    print(f"Total rows: {summary['row_count'].sum():,}")
    print(f"Total size: {summary['file_size_mb'].sum():.2f} MB")

    # Load all data
    print("\n" + "=" * 70)
    print("LOADING ALL DATA")
    print("=" * 70)
    df = loader.load_all_files()

    if df is not None:
        print(f"\nFinal shape: {df.shape}")
        print(f"\nDate range: {df['collection_date'].min()} to {df['collection_date'].max()}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())

        print(f"\nData types:")
        print(df.dtypes)

        print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
