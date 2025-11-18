"""
Cross-file deduplication for car advertisements
Handles duplicate ads across multiple months of data
Tracks price changes and ad evolution over time
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import PROCESSED_DATA_DIR
from src.utils.logger import get_logger, log_execution_time, log_dataframe_info
from src.utils.checkpoint_manager import DataFrameCheckpointManager


logger = get_logger(__name__)


class AdDeduplicator:
    """
    Deduplicates car advertisements across multiple data collection dates
    Identifies repriced ads and tracks temporal changes
    """

    def __init__(self):
        self.logger = logger

    @log_execution_time
    def deduplicate_within_file(self, df, keep='last'):
        """
        Remove duplicates within a single dataset

        Args:
            df: DataFrame with car ads
            keep: Which duplicate to keep ('first', 'last', False)

        Returns:
            Deduplicated DataFrame
        """
        initial_count = len(df)

        # Deduplicate by ad ID
        if 'id' in df.columns:
            df_dedup = df.drop_duplicates(subset=['id'], keep=keep)
        else:
            self.logger.warning("No 'id' column found, cannot deduplicate")
            return df

        removed_count = initial_count - len(df_dedup)
        self.logger.info(f"Removed {removed_count:,} duplicate ads ({removed_count/initial_count*100:.1f}%)")

        return df_dedup

    @log_execution_time
    def deduplicate_across_files(self, df, strategy='keep_latest'):
        """
        Deduplicate ads that appear in multiple months

        Args:
            df: Combined DataFrame with collection_date column
            strategy: 'keep_latest', 'keep_earliest', 'track_changes'

        Returns:
            Deduplicated DataFrame (or tracking DataFrame if strategy='track_changes')
        """
        if 'id' not in df.columns:
            self.logger.error("No 'id' column found")
            return df

        if 'collection_date' not in df.columns:
            self.logger.warning("No 'collection_date' column, cannot track across files")
            return self.deduplicate_within_file(df)

        initial_count = len(df)
        unique_ids = df['id'].nunique()
        duplicate_ids = initial_count - unique_ids

        self.logger.info(f"Total ads: {initial_count:,}")
        self.logger.info(f"Unique ad IDs: {unique_ids:,}")
        self.logger.info(f"Duplicate appearances: {duplicate_ids:,}")

        if strategy == 'keep_latest':
            # Sort by collection date and keep last appearance of each ID
            df_sorted = df.sort_values('collection_date')
            df_dedup = df_sorted.drop_duplicates(subset=['id'], keep='last')

            self.logger.info(f"Kept latest appearance of each ad")
            self.logger.info(f"Result: {len(df_dedup):,} unique ads")

            return df_dedup

        elif strategy == 'keep_earliest':
            # Sort by collection date and keep first appearance of each ID
            df_sorted = df.sort_values('collection_date')
            df_dedup = df_sorted.drop_duplicates(subset=['id'], keep='first')

            self.logger.info(f"Kept earliest appearance of each ad")
            self.logger.info(f"Result: {len(df_dedup):,} unique ads")

            return df_dedup

        elif strategy == 'track_changes':
            # Create time series tracking DataFrame
            return self.create_time_series_tracking(df)

        else:
            self.logger.error(f"Unknown strategy: {strategy}")
            return df

    @log_execution_time
    def create_time_series_tracking(self, df):
        """
        Create a DataFrame tracking ad evolution over time

        For ads that appear multiple times:
        - Track price changes
        - Calculate days on market
        - Flag repricing events

        Args:
            df: Combined DataFrame with collection_date

        Returns:
            Time series tracking DataFrame
        """
        if 'id' not in df.columns or 'collection_date' not in df.columns:
            self.logger.error("Missing required columns: id, collection_date")
            return df

        # Sort by ID and collection date
        df_sorted = df.sort_values(['id', 'collection_date'])

        # Group by ad ID
        grouped = df_sorted.groupby('id')

        tracking_data = []

        self.logger.info("Creating time series tracking...")

        for ad_id, group in grouped:
            if len(group) == 1:
                # Ad appeared only once
                row = group.iloc[0].to_dict()
                row['appearances'] = 1
                row['first_seen'] = row['collection_date']
                row['last_seen'] = row['collection_date']
                row['days_on_market'] = 0
                row['price_changes'] = 0
                row['price_change_amount'] = 0
                row['price_change_pct'] = 0
                row['initial_price'] = row.get('price_in_mil', None)
                row['final_price'] = row.get('price_in_mil', None)
                tracking_data.append(row)

            else:
                # Ad appeared multiple times
                first_row = group.iloc[0]
                last_row = group.iloc[-1]

                row = last_row.to_dict()  # Use latest version as base
                row['appearances'] = len(group)
                row['first_seen'] = group['collection_date'].min()
                row['last_seen'] = group['collection_date'].max()

                # Calculate days on market
                days = (row['last_seen'] - row['first_seen']).days
                row['days_on_market'] = days

                # Track price changes
                if 'price_in_mil' in group.columns:
                    prices = group['price_in_mil'].dropna()
                    if len(prices) > 1:
                        initial_price = prices.iloc[0]
                        final_price = prices.iloc[-1]
                        price_change = final_price - initial_price
                        price_change_pct = (price_change / initial_price * 100) if initial_price > 0 else 0

                        row['initial_price'] = initial_price
                        row['final_price'] = final_price
                        row['price_changes'] = (prices.diff().fillna(0) != 0).sum() - 1
                        row['price_change_amount'] = price_change
                        row['price_change_pct'] = price_change_pct
                    else:
                        row['initial_price'] = prices.iloc[0] if len(prices) > 0 else None
                        row['final_price'] = prices.iloc[0] if len(prices) > 0 else None
                        row['price_changes'] = 0
                        row['price_change_amount'] = 0
                        row['price_change_pct'] = 0

                tracking_data.append(row)

        tracking_df = pd.DataFrame(tracking_data)

        self.logger.info(f"Created tracking for {len(tracking_df):,} unique ads")
        self.logger.info(f"  Ads with multiple appearances: {(tracking_df['appearances'] > 1).sum():,}")
        self.logger.info(f"  Ads with price changes: {(tracking_df['price_changes'] > 0).sum():,}")

        return tracking_df

    def get_duplicate_summary(self, df):
        """
        Get summary statistics about duplicates

        Args:
            df: DataFrame with id and collection_date columns

        Returns:
            Dictionary with summary statistics
        """
        if 'id' not in df.columns:
            return {'error': 'No id column found'}

        total_ads = len(df)
        unique_ids = df['id'].nunique()
        duplicate_count = total_ads - unique_ids

        # Count appearances per ad
        appearance_counts = df['id'].value_counts()

        summary = {
            'total_ads': total_ads,
            'unique_ads': unique_ids,
            'duplicate_appearances': duplicate_count,
            'duplication_rate_pct': (duplicate_count / total_ads * 100) if total_ads > 0 else 0,
            'max_appearances': appearance_counts.max(),
            'avg_appearances': appearance_counts.mean(),
            'ads_appearing_once': (appearance_counts == 1).sum(),
            'ads_appearing_multiple': (appearance_counts > 1).sum()
        }

        # Top duplicated ads
        top_duplicates = appearance_counts.head(10)
        summary['top_duplicated_ids'] = top_duplicates.to_dict()

        return summary

    def analyze_repricing_behavior(self, tracking_df):
        """
        Analyze how sellers reprice their ads

        Args:
            tracking_df: Time series tracking DataFrame

        Returns:
            Dictionary with repricing analysis
        """
        if 'price_changes' not in tracking_df.columns:
            return {'error': 'No price_changes column found'}

        # Filter to ads with price changes
        repriced = tracking_df[tracking_df['price_changes'] > 0].copy()

        if len(repriced) == 0:
            return {'total_repriced_ads': 0}

        analysis = {
            'total_repriced_ads': len(repriced),
            'repricing_rate_pct': (len(repriced) / len(tracking_df) * 100),
            'avg_price_change_pct': repriced['price_change_pct'].mean(),
            'median_price_change_pct': repriced['price_change_pct'].median(),
            'avg_days_on_market': repriced['days_on_market'].mean(),
            'median_days_on_market': repriced['days_on_market'].median(),
            'price_increased_count': (repriced['price_change_amount'] > 0).sum(),
            'price_decreased_count': (repriced['price_change_amount'] < 0).sum(),
            'price_unchanged_count': (repriced['price_change_amount'] == 0).sum()
        }

        return analysis


# Helper functions
def deduplicate_data(df, strategy='keep_latest', checkpoint=True):
    """
    Quick deduplication function

    Args:
        df: DataFrame to deduplicate
        strategy: Deduplication strategy
        checkpoint: Save checkpoint

    Returns:
        Deduplicated DataFrame
    """
    dedup = AdDeduplicator()

    if strategy == 'track_changes':
        result = dedup.create_time_series_tracking(df)
    else:
        result = dedup.deduplicate_across_files(df, strategy=strategy)

    # Save checkpoint
    if checkpoint:
        checkpoint_mgr = DataFrameCheckpointManager(
            PROCESSED_DATA_DIR,
            f"deduplicated_{strategy}",
            format='parquet'
        )
        checkpoint_mgr.save(result, metadata={'strategy': strategy})

    return result


# Example usage
if __name__ == "__main__":
    from src.utils.logger import setup_logger
    from src.data.loader import CarDataLoader

    # Setup logging
    setup_logger("deduplicator", level='INFO')

    # Load data
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    loader = CarDataLoader()
    df = loader.load_all_files()

    if df is not None:
        # Create deduplicator
        dedup = AdDeduplicator()

        # Get duplicate summary
        print("\n" + "=" * 70)
        print("DUPLICATE SUMMARY")
        print("=" * 70)
        summary = dedup.get_duplicate_summary(df)
        for key, value in summary.items():
            if key != 'top_duplicated_ids':
                print(f"{key}: {value}")

        # Deduplicate - keep latest
        print("\n" + "=" * 70)
        print("DEDUPLICATION - KEEP LATEST")
        print("=" * 70)
        df_latest = dedup.deduplicate_across_files(df, strategy='keep_latest')
        print(f"Result shape: {df_latest.shape}")

        # Create time series tracking
        print("\n" + "=" * 70)
        print("TIME SERIES TRACKING")
        print("=" * 70)
        df_tracking = dedup.create_time_series_tracking(df)
        print(f"Tracking shape: {df_tracking.shape}")

        # Analyze repricing
        print("\n" + "=" * 70)
        print("REPRICING ANALYSIS")
        print("=" * 70)
        repricing = dedup.analyze_repricing_behavior(df_tracking)
        for key, value in repricing.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
