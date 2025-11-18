"""
Feature engineering for car price prediction
Creates derived features from raw data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import CURRENT_YEAR, STANDARD_COLORS, ENCODING_STRATEGIES, PROCESSED_DATA_DIR
from src.utils.logger import get_logger, log_execution_time


logger = get_logger(__name__)


class FeatureEngineer:
    """Creates features for machine learning models"""

    def __init__(self):
        self.logger = logger
        self.encoders = {}

    @log_execution_time
    def create_temporal_features(self, df):
        """Create time-based features"""
        self.logger.info("Creating temporal features...")

        # Car age
        if 'manufacture_year' in df.columns:
            df['car_age'] = CURRENT_YEAR - df['manufacture_year']

        # Import delay
        if 'import_year' in df.columns and 'manufacture_year' in df.columns:
            df['import_delay'] = df['import_year'] - df['manufacture_year']

        # Mileage per year
        if 'mileage' in df.columns and 'car_age' in df.columns:
            df['mileage_per_year'] = df['mileage'] / (df['car_age'] + 1)  # +1 to avoid division by zero

        # Is brand new (0 mileage)
        if 'mileage' in df.columns:
            df['is_brand_new'] = (df['mileage'] == 0).astype(int)

        # Collection month (seasonality)
        if 'collection_date' in df.columns:
            df['collection_month'] = pd.to_datetime(df['collection_date']).dt.month
            df['collection_quarter'] = pd.to_datetime(df['collection_date']).dt.quarter

        return df

    @log_execution_time
    def create_categorical_features(self, df):
        """Create categorical features"""
        self.logger.info("Creating categorical features...")

        # Is bright color (non-standard)
        if 'color' in df.columns:
            df['is_bright_color'] = ~df['color'].str.lower().isin([c.lower() for c in STANDARD_COLORS])
            df['is_bright_color'] = df['is_bright_color'].astype(int)

        # Is automatic transmission
        if 'transmission' in df.columns:
            df['is_automatic'] = df['transmission'].str.lower().isin(['автомат', 'automatic', 'auto']).astype(int)

        # Is diesel/electric/hybrid
        if 'engine_type' in df.columns:
            df['is_diesel'] = df['engine_type'].str.lower().str.contains('diesel|дизель', na=False).astype(int)
            df['is_electric'] = df['engine_type'].str.lower().str.contains('electric|цахилгаан', na=False).astype(int)
            df['is_hybrid'] = df['engine_type'].str.lower().str.contains('hybrid|эрлийз', na=False).astype(int)

        # Is 4WD/AWD
        if 'drive' in df.columns:
            df['is_4wd'] = df['drive'].str.lower().isin(['4wd', 'awd', '4x4']).astype(int)

        return df

    @log_execution_time
    def create_price_segments(self, df):
        """Create price and mileage segments"""
        self.logger.info("Creating price/mileage segments...")

        # Price segments
        if 'price_in_mil' in df.columns:
            df['price_segment'] = pd.cut(df['price_in_mil'],
                                         bins=[0, 20, 40, 60, 100, float('inf')],
                                         labels=['budget', 'mid', 'premium', 'luxury', 'ultra_luxury'])

        # Mileage segments
        if 'mileage' in df.columns:
            df['mileage_segment'] = pd.cut(df['mileage'],
                                           bins=[0, 50000, 100000, 200000, float('inf')],
                                           labels=['low', 'medium', 'high', 'very_high'])

        # Age segments
        if 'car_age' in df.columns:
            df['age_segment'] = pd.cut(df['car_age'],
                                       bins=[-1, 3, 7, 10, float('inf')],
                                       labels=['new', 'recent', 'old', 'very_old'])

        return df

    @log_execution_time
    def engineer_features(self, df):
        """Complete feature engineering pipeline"""
        self.logger.info(f"Starting feature engineering on {len(df):,} ads")

        df = self.create_temporal_features(df)
        df = self.create_categorical_features(df)
        df = self.create_price_segments(df)

        self.logger.info(f"Feature engineering complete. Shape: {df.shape}")

        return df


# Quick function
def engineer_features(df):
    """Quick feature engineering"""
    fe = FeatureEngineer()
    return fe.engineer_features(df)


if __name__ == "__main__":
    from src.utils.logger import setup_logger
    from src.data.loader import CarDataLoader
    from src.data.cleaner import DataCleaner

    setup_logger("feature_eng", level='INFO')

    loader = CarDataLoader()
    df = loader.load_all_files()

    if df is not None:
        cleaner = DataCleaner()
        df_clean = cleaner.clean_data(df)

        fe = FeatureEngineer()
        df_featured = fe.engineer_features(df_clean)

        print(f"\nFinal shape: {df_featured.shape}")
        print(f"\nNew columns created:")
        original_cols = df.columns
        new_cols = [c for c in df_featured.columns if c not in original_cols]
        for col in new_cols:
            print(f"  - {col}")
