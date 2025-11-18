"""
Data cleaning module for car advertisements
Handles outliers, missing values, and data quality issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy import stats

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    PROCESSED_DATA_DIR,
    PRICE_OUTLIER_METHOD, PRICE_IQR_MULTIPLIER, PRICE_ZSCORE_THRESHOLD, PRICE_PERCENTILE_RANGE,
    MILEAGE_OUTLIER_METHOD, MILEAGE_IQR_MULTIPLIER, MILEAGE_ZSCORE_THRESHOLD,
    MIN_PRICE, MAX_PRICE, MIN_MILEAGE, MAX_MILEAGE,
    MIN_MANUFACTURE_YEAR, MAX_MANUFACTURE_YEAR, MAX_MISSING_RATIO
)
from src.utils.logger import get_logger, log_execution_time, log_dataframe_info
from src.utils.checkpoint_manager import DataFrameCheckpointManager


logger = get_logger(__name__)


class DataCleaner:
    """
    Cleans car advertisement data
    Removes outliers, handles missing values, validates data ranges
    """

    def __init__(self):
        self.logger = logger
        self.cleaning_report = {}

    def detect_outliers_iqr(self, series, multiplier=1.5):
        """
        Detect outliers using IQR method

        Args:
            series: pandas Series
            multiplier: IQR multiplier (default 1.5, use 3.0 for more lenient)

        Returns:
            Boolean mask (True = outlier)
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outliers = (series < lower_bound) | (series > upper_bound)

        return outliers

    def detect_outliers_zscore(self, series, threshold=3):
        """
        Detect outliers using Z-score method

        Args:
            series: pandas Series
            threshold: Z-score threshold (typically 3)

        Returns:
            Boolean mask (True = outlier)
        """
        z_scores = np.abs(stats.zscore(series.dropna()))
        outliers = pd.Series(False, index=series.index)
        outliers.loc[series.dropna().index] = z_scores > threshold

        return outliers

    def detect_outliers_percentile(self, series, percentile_range=(0.5, 99.5)):
        """
        Detect outliers using percentile method

        Args:
            series: pandas Series
            percentile_range: Tuple of (lower, upper) percentiles

        Returns:
            Boolean mask (True = outlier)
        """
        lower_bound = series.quantile(percentile_range[0] / 100)
        upper_bound = series.quantile(percentile_range[1] / 100)

        outliers = (series < lower_bound) | (series > upper_bound)

        return outliers

    @log_execution_time
    def remove_price_outliers(self, df, method=None, remove=True):
        """
        Detect and optionally remove price outliers

        Args:
            df: DataFrame
            method: 'iqr', 'zscore', or 'percentile' (uses config default if None)
            remove: If True, remove outliers; if False, just flag them

        Returns:
            Cleaned DataFrame (or DataFrame with outlier flags)
        """
        if 'price_in_mil' not in df.columns:
            self.logger.warning("No price_in_mil column found")
            return df

        method = method if method else PRICE_OUTLIER_METHOD

        # First, apply hard limits
        valid_range = (df['price_in_mil'] >= MIN_PRICE) & (df['price_in_mil'] <= MAX_PRICE)
        invalid_count = (~valid_range).sum()

        self.logger.info(f"Price range validation: {invalid_count:,} ads outside [{MIN_PRICE}, {MAX_PRICE}] range")

        # Detect outliers using selected method
        price_series = df.loc[valid_range, 'price_in_mil']

        if method == 'iqr':
            outliers_mask = self.detect_outliers_iqr(price_series, PRICE_IQR_MULTIPLIER)
        elif method == 'zscore':
            outliers_mask = self.detect_outliers_zscore(price_series, PRICE_ZSCORE_THRESHOLD)
        elif method == 'percentile':
            outliers_mask = self.detect_outliers_percentile(price_series, PRICE_PERCENTILE_RANGE)
        else:
            self.logger.error(f"Unknown outlier method: {method}")
            return df

        outlier_count = outliers_mask.sum()
        self.logger.info(f"Detected {outlier_count:,} price outliers using {method} method")

        # Store statistics
        self.cleaning_report['price_outliers'] = {
            'method': method,
            'invalid_range': int(invalid_count),
            'outliers_detected': int(outlier_count),
            'total_removed': int(invalid_count + outlier_count)
        }

        if remove:
            # Remove both invalid range and outliers
            full_outliers_mask = ~valid_range
            full_outliers_mask.loc[valid_range] = outliers_mask

            df_clean = df[~full_outliers_mask].copy()
            self.logger.info(f"Removed {full_outliers_mask.sum():,} total price outliers")
            self.logger.info(f"Remaining ads: {len(df_clean):,}")

            return df_clean
        else:
            # Just flag outliers
            df['price_outlier'] = False
            df.loc[~valid_range, 'price_outlier'] = True
            df.loc[valid_range, 'price_outlier'] = outliers_mask

            return df

    @log_execution_time
    def remove_mileage_outliers(self, df, method=None, remove=True):
        """
        Detect and optionally remove mileage outliers

        Args:
            df: DataFrame
            method: 'iqr', 'zscore', or 'percentile'
            remove: If True, remove outliers; if False, just flag them

        Returns:
            Cleaned DataFrame
        """
        if 'mileage' not in df.columns:
            self.logger.warning("No mileage column found")
            return df

        method = method if method else MILEAGE_OUTLIER_METHOD

        # First, apply hard limits
        valid_range = (df['mileage'] >= MIN_MILEAGE) & (df['mileage'] <= MAX_MILEAGE)
        invalid_count = (~valid_range).sum()

        self.logger.info(f"Mileage range validation: {invalid_count:,} ads outside [{MIN_MILEAGE}, {MAX_MILEAGE}] range")

        # Detect outliers using selected method
        mileage_series = df.loc[valid_range, 'mileage']

        if method == 'iqr':
            outliers_mask = self.detect_outliers_iqr(mileage_series, MILEAGE_IQR_MULTIPLIER)
        elif method == 'zscore':
            outliers_mask = self.detect_outliers_zscore(mileage_series, MILEAGE_ZSCORE_THRESHOLD)
        elif method == 'percentile':
            outliers_mask = self.detect_outliers_percentile(mileage_series, PRICE_PERCENTILE_RANGE)
        else:
            self.logger.error(f"Unknown outlier method: {method}")
            return df

        outlier_count = outliers_mask.sum()
        self.logger.info(f"Detected {outlier_count:,} mileage outliers using {method} method")

        # Store statistics
        self.cleaning_report['mileage_outliers'] = {
            'method': method,
            'invalid_range': int(invalid_count),
            'outliers_detected': int(outlier_count),
            'total_removed': int(invalid_count + outlier_count)
        }

        if remove:
            # Remove both invalid range and outliers
            full_outliers_mask = ~valid_range
            full_outliers_mask.loc[valid_range] = outliers_mask

            df_clean = df[~full_outliers_mask].copy()
            self.logger.info(f"Removed {full_outliers_mask.sum():,} total mileage outliers")

            return df_clean
        else:
            # Just flag outliers
            df['mileage_outlier'] = False
            df.loc[~valid_range, 'mileage_outlier'] = True
            df.loc[valid_range, 'mileage_outlier'] = outliers_mask

            return df

    @log_execution_time
    def validate_year_ranges(self, df, remove=True):
        """
        Validate manufacture and import years

        Args:
            df: DataFrame
            remove: If True, remove invalid years

        Returns:
            Cleaned DataFrame
        """
        initial_count = len(df)

        if 'manufacture_year' in df.columns:
            valid_manufacture = (
                (df['manufacture_year'] >= MIN_MANUFACTURE_YEAR) &
                (df['manufacture_year'] <= MAX_MANUFACTURE_YEAR)
            ) | df['manufacture_year'].isna()

            invalid_count = (~valid_manufacture).sum()
            self.logger.info(f"Invalid manufacture years: {invalid_count:,}")

            if remove:
                df = df[valid_manufacture].copy()

        if 'import_year' in df.columns:
            valid_import = (
                (df['import_year'] >= df['manufacture_year']) &
                (df['import_year'] <= MAX_MANUFACTURE_YEAR)
            ) | df['import_year'].isna()

            invalid_count = (~valid_import).sum()
            self.logger.info(f"Invalid import years: {invalid_count:,}")

            if remove:
                df = df[valid_import].copy()

        removed_count = initial_count - len(df)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count:,} ads with invalid years")

        return df

    @log_execution_time
    def handle_missing_values(self, df, strategy='smart'):
        """
        Handle missing values

        Args:
            df: DataFrame
            strategy: 'drop_columns', 'drop_rows', 'smart', 'impute'

        Returns:
            DataFrame with handled missing values
        """
        self.logger.info("Handling missing values...")

        # Log missing value summary
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        cols_with_missing = missing[missing > 0].sort_values(ascending=False)

        self.logger.info(f"Columns with missing values: {len(cols_with_missing)}")
        for col in cols_with_missing.index[:10]:  # Show top 10
            self.logger.info(f"  {col}: {missing[col]:,} ({missing_pct[col]:.1f}%)")

        if strategy == 'drop_columns':
            # Drop columns with too many missing values
            cols_to_drop = missing_pct[missing_pct > MAX_MISSING_RATIO * 100].index.tolist()
            if cols_to_drop:
                self.logger.info(f"Dropping {len(cols_to_drop)} columns: {cols_to_drop}")
                df = df.drop(columns=cols_to_drop)

        elif strategy == 'drop_rows':
            # Drop rows with any missing values in critical columns
            critical_cols = ['id', 'brand', 'mark', 'price_in_mil', 'mileage', 'manufacture_year']
            available_critical = [c for c in critical_cols if c in df.columns]

            initial_count = len(df)
            df = df.dropna(subset=available_critical)
            removed_count = initial_count - len(df)
            self.logger.info(f"Removed {removed_count:,} rows with missing critical values")

        elif strategy == 'smart':
            # Smart strategy: drop columns with >50% missing, impute important ones
            # Drop high-missing columns
            cols_to_drop = missing_pct[missing_pct > MAX_MISSING_RATIO * 100].index.tolist()
            if cols_to_drop:
                self.logger.info(f"Dropping {len(cols_to_drop)} high-missing columns")
                df = df.drop(columns=cols_to_drop)

            # Drop rows missing critical values
            critical_cols = ['id', 'brand', 'mark', 'price_in_mil']
            available_critical = [c for c in critical_cols if c in df.columns]

            if available_critical:
                initial_count = len(df)
                df = df.dropna(subset=available_critical)
                removed_count = initial_count - len(df)
                if removed_count > 0:
                    self.logger.info(f"Removed {removed_count:,} rows with missing critical values")

        return df

    @log_execution_time
    def clean_data(self, df, remove_outliers=True, handle_missing=True, checkpoint=True):
        """
        Complete data cleaning pipeline

        Args:
            df: Raw DataFrame
            remove_outliers: Remove price and mileage outliers
            handle_missing: Handle missing values
            checkpoint: Save checkpoint

        Returns:
            Cleaned DataFrame
        """
        initial_count = len(df)
        self.logger.info(f"Starting data cleaning: {initial_count:,} ads")

        # Reset cleaning report
        self.cleaning_report = {'initial_count': initial_count}

        # Remove duplicates (if not already done)
        if 'id' in df.columns:
            df = df.drop_duplicates(subset=['id'], keep='last')
            dup_removed = initial_count - len(df)
            if dup_removed > 0:
                self.logger.info(f"Removed {dup_removed:,} duplicates")
                self.cleaning_report['duplicates_removed'] = int(dup_removed)

        # Validate year ranges
        df = self.validate_year_ranges(df, remove=True)

        # Remove outliers
        if remove_outliers:
            df = self.remove_price_outliers(df, remove=True)
            df = self.remove_mileage_outliers(df, remove=True)

        # Handle missing values
        if handle_missing:
            df = self.handle_missing_values(df, strategy='smart')

        final_count = len(df)
        total_removed = initial_count - final_count
        retention_rate = (final_count / initial_count) * 100

        self.logger.info(f"Data cleaning complete!")
        self.logger.info(f"  Initial: {initial_count:,} ads")
        self.logger.info(f"  Final: {final_count:,} ads")
        self.logger.info(f"  Removed: {total_removed:,} ads ({100-retention_rate:.1f}%)")
        self.logger.info(f"  Retention: {retention_rate:.1f}%")

        self.cleaning_report['final_count'] = int(final_count)
        self.cleaning_report['total_removed'] = int(total_removed)
        self.cleaning_report['retention_rate'] = float(retention_rate)

        # Save checkpoint
        if checkpoint:
            checkpoint_mgr = DataFrameCheckpointManager(
                PROCESSED_DATA_DIR,
                "cleaned_data",
                format='parquet'
            )
            checkpoint_mgr.save(df, metadata=self.cleaning_report)

        return df

    def get_cleaning_report(self):
        """Get summary report of cleaning operations"""
        return self.cleaning_report


# Helper function
def clean_car_data(df, **kwargs):
    """Quick cleaning function"""
    cleaner = DataCleaner()
    return cleaner.clean_data(df, **kwargs)


# Example usage
if __name__ == "__main__":
    from src.utils.logger import setup_logger
    from src.data.loader import CarDataLoader

    # Setup logging
    setup_logger("cleaner", level='INFO')

    # Load data
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    loader = CarDataLoader()
    df = loader.load_all_files()

    if df is not None:
        # Clean data
        print("\n" + "=" * 70)
        print("CLEANING DATA")
        print("=" * 70)

        cleaner = DataCleaner()
        df_clean = cleaner.clean_data(df)

        # Show report
        print("\n" + "=" * 70)
        print("CLEANING REPORT")
        print("=" * 70)
        report = cleaner.get_cleaning_report()
        for key, value in report.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
