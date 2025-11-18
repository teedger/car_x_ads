"""
Configuration file for Car Price Prediction Project
Centralized settings for paths, parameters, and constants
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MODELS_DIR = OUTPUTS_DIR / "models"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR,
                  FIGURES_DIR, REPORTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA FILES
# ============================================================================
# Data file pattern for car CSV files
DATA_FILE_PATTERN = "car_cleaned_*.csv"

# Processed data file names
MERGED_DATA_FILE = PROCESSED_DATA_DIR / "merged_all_months.csv"
UNIQUE_ADS_FILE = PROCESSED_DATA_DIR / "unique_ads.csv"
TIME_SERIES_FILE = PROCESSED_DATA_DIR / "time_series_ads.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_data.csv"
FEATURE_ENGINEERED_FILE = PROCESSED_DATA_DIR / "featured_data.csv"

# ============================================================================
# DATA SCHEMA
# ============================================================================
# Expected columns in the CSV files
EXPECTED_COLUMNS = [
    'id', 'ad_title', 'brand', 'mark', 'price_in_mil', 'currency',
    'ad_link', 'ad_date', 'location', 'city/province', 'district',
    'neighbourhood', 'engine_capacity', 'transmission', 'steering',
    'type', 'color', 'manufacture_year', 'import_year', 'engine_type',
    'inter_color', 'leasing', 'drive', 'mileage', 'condition', 'doors'
]

# Column data types
COLUMN_DTYPES = {
    'id': 'int64',
    'ad_title': 'str',
    'brand': 'str',
    'mark': 'str',
    'price_in_mil': 'float64',
    'currency': 'str',
    'ad_link': 'str',
    'location': 'str',
    'city/province': 'str',
    'district': 'str',
    'neighbourhood': 'str',
    'engine_capacity': 'str',
    'transmission': 'str',
    'steering': 'str',
    'type': 'str',
    'color': 'str',
    'manufacture_year': 'float64',
    'import_year': 'float64',
    'engine_type': 'str',
    'inter_color': 'str',
    'leasing': 'str',
    'drive': 'str',
    'mileage': 'int64',
    'condition': 'str',
    'doors': 'float64'
}

# Date columns to parse
DATE_COLUMNS = ['ad_date']

# ============================================================================
# DATA CLEANING PARAMETERS
# ============================================================================
# Outlier detection parameters
PRICE_OUTLIER_METHOD = 'iqr'  # 'iqr', 'zscore', or 'percentile'
PRICE_IQR_MULTIPLIER = 3.0  # For IQR method
PRICE_ZSCORE_THRESHOLD = 3.5  # For Z-score method
PRICE_PERCENTILE_RANGE = (0.5, 99.5)  # For percentile method

MILEAGE_OUTLIER_METHOD = 'iqr'
MILEAGE_IQR_MULTIPLIER = 3.0
MILEAGE_ZSCORE_THRESHOLD = 3.5

# Missing value thresholds
MAX_MISSING_RATIO = 0.5  # Drop columns with >50% missing

# Valid ranges
MIN_PRICE = 0.5  # Million MNT
MAX_PRICE = 500  # Million MNT
MIN_MILEAGE = 0  # km
MAX_MILEAGE = 1000000  # km
MIN_MANUFACTURE_YEAR = 1980
MAX_MANUFACTURE_YEAR = 2026  # Future year for upcoming models

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
CURRENT_YEAR = 2025  # Reference year for calculations

# Standard colors (non-bright)
STANDARD_COLORS = ['black', 'white', 'silver', 'grey', 'gray', 'хар', 'цагаан', 'саарал']

# Categorical encoding strategies
ENCODING_STRATEGIES = {
    'brand': 'target',  # target, onehot, label
    'mark': 'target',
    'type': 'onehot',
    'color': 'label',
    'transmission': 'onehot',
    'fuel_type': 'onehot',
    'district': 'target'
}

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42
VALIDATION_SIZE = 0.15  # From training set

# Cross-validation
CV_FOLDS = 5
CV_SCORING = 'r2'  # or 'neg_mean_squared_error'

# Model selection
MODELS_TO_TRAIN = [
    'linear_regression',
    'ridge',
    'lasso',
    'decision_tree',
    'random_forest',
    'gradient_boosting',
    'xgboost',
    'lightgbm',
    'catboost'
]

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

XGBOOST_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

LIGHTGBM_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100]
}

CATBOOST_PARAMS = {
    'iterations': [100, 200, 300],
    'depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1]
}

# Target accuracy
TARGET_R2_SCORE = 0.98

# ============================================================================
# CHECKPOINT SETTINGS
# ============================================================================
CHECKPOINT_FREQUENCY = 1000  # Save checkpoint every N iterations
AUTO_SAVE_ENABLED = True
CHECKPOINT_PREFIX = "checkpoint"

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
# Plot style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'  # png, jpg, svg, pdf

# Plot sizes
SMALL_PLOT_SIZE = (10, 6)
MEDIUM_PLOT_SIZE = (12, 8)
LARGE_PLOT_SIZE = (16, 10)
WIDE_PLOT_SIZE = (18, 6)

# Color palettes
COLOR_PALETTE = 'Set2'
SEQUENTIAL_PALETTE = 'Blues'
DIVERGING_PALETTE = 'RdYlGn'

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
# For question analysis
TOP_N_BRANDS = 15  # Show top N brands in charts
TOP_N_MARKS = 20  # Show top N marks in charts
TOP_N_DISTRICTS = 10  # Show top N districts

# Time series parameters
SEASONALITY_PERIOD = 12  # months
FORECAST_HORIZON = 30  # days

# ============================================================================
# LOGGING SETTINGS
# ============================================================================
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = PROJECT_ROOT / 'car_analysis.log'

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================
# Parallel processing
N_JOBS = -1  # Use all available cores (-1)
CHUNK_SIZE = 10000  # For chunked data processing
LOW_MEMORY = True  # Use low memory mode for large files

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_data_files():
    """Get list of all car data CSV files"""
    from glob import glob
    files = sorted(glob(str(RAW_DATA_DIR / DATA_FILE_PATTERN)))
    return files

def get_checkpoint_path(name):
    """Get checkpoint file path"""
    return CHECKPOINT_DIR / f"{CHECKPOINT_PREFIX}_{name}.pkl"

def get_figure_path(name):
    """Get figure file path"""
    return FIGURES_DIR / f"{name}.{FIGURE_FORMAT}"

def get_report_path(name):
    """Get report file path"""
    return REPORTS_DIR / f"{name}.txt"

def get_model_path(name):
    """Get model file path"""
    return MODELS_DIR / f"{name}.pkl"


if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Number of data files found: {len(get_data_files())}")
