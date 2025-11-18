#!/usr/bin/env python3
"""
Complete Analysis Pipeline for Car Price Prediction
Runs the full data processing and analysis workflow

Usage:
    python run_pipeline.py                    # Run full pipeline
    python run_pipeline.py --skip-load        # Skip data loading
    python run_pipeline.py --skip-ml          # Skip ML training
    python run_pipeline.py --quick            # Quick run (subset of data)
"""

import argparse
import sys
from pathlib import Path
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.logger import setup_logger
from src.data.loader import CarDataLoader
from src.data.deduplicator import AdDeduplicator
from src.data.cleaner import DataCleaner
from src.data.feature_engineering import FeatureEngineer
from src.analysis.eda_11_questions import EDAAnalyzer


def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_stats(df, name="Dataset"):
    """Print dataset statistics"""
    print(f"\n{name} Statistics:")
    print(f"  Shape: {df.shape}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    if 'collection_date' in df.columns:
        print(f"  Date range: {df['collection_date'].min()} to {df['collection_date'].max()}")


def run_data_loading(skip=False):
    """Step 1: Load data from multiple CSV files"""
    if skip:
        print("⏭️  Skipping data loading (loading from checkpoint)")
        return None

    print_header("STEP 1: DATA LOADING")

    loader = CarDataLoader()

    # Show file summary
    print("📁 Scanning data files...")
    summary = loader.get_file_summary()

    if len(summary) == 0:
        print("\n❌ ERROR: No data files found!")
        print("   Please place CSV files in data/raw/ directory")
        print("   Expected format: car_cleaned_YYYYMMDD.csv")
        sys.exit(1)

    print(f"\n✓ Found {len(summary)} files")
    print(f"  Total rows: {summary['row_count'].sum():,}")
    print(f"  Total size: {summary['file_size_mb'].sum():.2f} MB")
    print(f"  Date range: {summary['collection_date'].min()} to {summary['collection_date'].max()}")

    # Load all files
    print("\n📥 Loading all data files...")
    df = loader.load_all_files(use_checkpoint=True)

    print_stats(df, "Loaded Data")
    print("\n✅ Data loading complete")

    return df


def run_deduplication(df):
    """Step 2: Deduplicate ads across files"""
    print_header("STEP 2: DEDUPLICATION")

    dedup = AdDeduplicator()

    # Get duplicate summary
    print("🔍 Analyzing duplicates...")
    dup_summary = dedup.get_duplicate_summary(df)

    print(f"\n📊 Duplicate Summary:")
    print(f"  Total ads: {dup_summary['total_ads']:,}")
    print(f"  Unique ad IDs: {dup_summary['unique_ads']:,}")
    print(f"  Duplicate appearances: {dup_summary['duplicate_appearances']:,}")
    print(f"  Duplication rate: {dup_summary['duplication_rate_pct']:.1f}%")

    # Deduplicate
    print("\n🗑️  Removing duplicates (keeping latest)...")
    df_dedup = dedup.deduplicate_across_files(df, strategy='keep_latest')

    print_stats(df_dedup, "Deduplicated Data")
    print("\n✅ Deduplication complete")

    return df_dedup


def run_cleaning(df):
    """Step 3: Clean data"""
    print_header("STEP 3: DATA CLEANING")

    cleaner = DataCleaner()

    print("🧹 Cleaning data...")
    print("  - Validating year ranges")
    print("  - Removing price outliers")
    print("  - Removing mileage outliers")
    print("  - Handling missing values")

    df_clean = cleaner.clean_data(df, remove_outliers=True, handle_missing=True)

    # Show cleaning report
    report = cleaner.get_cleaning_report()
    print(f"\n📋 Cleaning Report:")
    print(f"  Initial count: {report['initial_count']:,}")
    print(f"  Final count: {report['final_count']:,}")
    print(f"  Total removed: {report['total_removed']:,}")
    print(f"  Retention rate: {report['retention_rate']:.1f}%")

    if 'price_outliers' in report:
        po = report['price_outliers']
        print(f"\n  Price outliers:")
        print(f"    Method: {po['method']}")
        print(f"    Removed: {po['total_removed']:,}")

    if 'mileage_outliers' in report:
        mo = report['mileage_outliers']
        print(f"\n  Mileage outliers:")
        print(f"    Method: {mo['method']}")
        print(f"    Removed: {mo['total_removed']:,}")

    print_stats(df_clean, "Cleaned Data")
    print("\n✅ Data cleaning complete")

    return df_clean


def run_feature_engineering(df):
    """Step 4: Engineer features"""
    print_header("STEP 4: FEATURE ENGINEERING")

    fe = FeatureEngineer()

    print("🔧 Engineering features...")
    print("  - Creating temporal features (car_age, mileage_per_year, etc.)")
    print("  - Creating categorical features (is_automatic, is_4wd, etc.)")
    print("  - Creating price/mileage segments")

    df_featured = fe.engineer_features(df)

    # Show new features
    original_cols = set(df.columns)
    new_cols = [c for c in df_featured.columns if c not in original_cols]

    print(f"\n📊 Created {len(new_cols)} new features:")
    for col in new_cols[:10]:  # Show first 10
        print(f"  - {col}")
    if len(new_cols) > 10:
        print(f"  ... and {len(new_cols) - 10} more")

    print_stats(df_featured, "Featured Data")
    print("\n✅ Feature engineering complete")

    return df_featured


def run_eda_analysis(df):
    """Step 5: Run EDA (11 questions)"""
    print_header("STEP 5: EXPLORATORY DATA ANALYSIS")

    analyzer = EDAAnalyzer(df)

    print("📊 Running analysis for 11 research questions...")
    print()

    questions = [
        "Q1: Most sold brands and marks",
        "Q2: Brands with highest mileage",
        "Q3: Manufacture years with highest mileage",
        "Q4: Monthly ad volume",
        "Q5: Ads by district (traffic comparison)",
        "Q6: Oldest cars by location",
        "Q7: Newest cars by location",
        "Q8: Mileage comparison (2015-2024 vs older)",
        "Q9: Bright colored cars by location",
        "Q10: Fuel type breakdown",
        "Q11: Car type breakdown"
    ]

    for i, question in enumerate(questions, 1):
        print(f"  {i}/11 - {question}...")

    results = analyzer.run_all_questions(save_results=True)

    print(f"\n✅ Analysis complete!")
    print(f"   Results saved to: outputs/reports/11_questions_results.json")

    # Show sample results
    print("\n📈 Sample Results:")
    if 'q1' in results:
        top_brand = list(results['q1']['top_brands'].keys())[0]
        top_count = results['q1']['top_brands'][top_brand]
        print(f"  Most sold brand: {top_brand} ({top_count:,} ads)")

    if 'q4' in results and 'avg_monthly_ads' in results['q4']:
        print(f"  Average monthly ads: {results['q4']['avg_monthly_ads']:.0f}")

    return results


def run_ml_training(df, quick=False):
    """Step 6: Train ML models"""
    print_header("STEP 6: MACHINE LEARNING")

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        import numpy as np
    except ImportError:
        print("⚠️  ML libraries not installed. Skipping ML training.")
        print("   Install with: pip install scikit-learn")
        return None

    print("🤖 Preparing data for machine learning...")

    # Select features
    feature_cols = ['car_age', 'mileage', 'mileage_per_year', 'is_automatic',
                    'is_4wd', 'is_brand_new']

    # Filter available features
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Add encoded categorical features
    for cat in ['brand', 'type', 'transmission']:
        if cat in df.columns:
            df[f'{cat}_encoded'] = pd.Categorical(df[cat]).codes
            feature_cols.append(f'{cat}_encoded')

    # Prepare X and y
    ml_df = df.dropna(subset=feature_cols + ['price_in_mil'])

    if quick:
        print("⚡ Quick mode: Using 10,000 samples")
        ml_df = ml_df.sample(n=min(10000, len(ml_df)), random_state=42)

    X = ml_df[feature_cols]
    y = ml_df['price_in_mil']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    print(f"  Features: {len(feature_cols)}")

    # Train Random Forest
    print("\n🌲 Training Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=100 if not quick else 50,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    start_time = time.time()
    rf_model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Evaluate
    y_pred = rf_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n📊 Model Performance:")
    print(f"  R² Score: {r2:.4f} ({r2*100:.2f}% accuracy)")
    print(f"  MAE: {mae:.2f} million MNT")
    print(f"  RMSE: {rmse:.2f} million MNT")
    print(f"  Training time: {train_time:.2f}s")

    if r2 >= 0.98:
        print("\n🎉 TARGET ACHIEVED! R² >= 0.98")
    elif r2 >= 0.95:
        print("\n✅ GOOD PERFORMANCE! R² >= 0.95")
    elif r2 >= 0.90:
        print("\n⚠️  NEEDS IMPROVEMENT. R² >= 0.90 but < 0.95")
    else:
        print("\n❌ POOR PERFORMANCE. R² < 0.90")
        print("   Try: More feature engineering, hyperparameter tuning, or different models")

    # Save model
    import joblib
    from config import MODELS_DIR
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "random_forest_model.pkl"
    joblib.dump(rf_model, model_path)
    print(f"\n💾 Model saved to: {model_path}")

    print("\n✅ ML training complete")

    return {'r2': r2, 'mae': mae, 'rmse': rmse}


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Car Price Prediction Pipeline')
    parser.add_argument('--skip-load', action='store_true', help='Skip data loading')
    parser.add_argument('--skip-ml', action='store_true', help='Skip ML training')
    parser.add_argument('--quick', action='store_true', help='Quick run (subset of data)')
    args = parser.parse_args()

    # Setup logging
    setup_logger('pipeline', level='INFO')

    print("\n" + "=" * 80)
    print("  🚗 CAR PRICE PREDICTION PIPELINE")
    print("  Mongolia Second-Hand Car Market Analysis")
    print("=" * 80)

    start_time = time.time()

    try:
        # Step 1: Load data
        df = run_data_loading(skip=args.skip_load)

        if df is None:
            # Try to load from checkpoint
            from config import PROCESSED_DATA_DIR
            from src.utils.checkpoint_manager import DataFrameCheckpointManager

            checkpoint_mgr = DataFrameCheckpointManager(
                PROCESSED_DATA_DIR, "merged_data", format='parquet'
            )
            df, _ = checkpoint_mgr.load()

            if df is None:
                print("\n❌ No data available. Please run without --skip-load first.")
                sys.exit(1)

        # Step 2: Deduplicate
        df_dedup = run_deduplication(df)

        # Step 3: Clean
        df_clean = run_cleaning(df_dedup)

        # Step 4: Feature engineering
        df_featured = run_feature_engineering(df_clean)

        # Step 5: EDA
        results = run_eda_analysis(df_featured)

        # Step 6: ML training
        if not args.skip_ml:
            ml_results = run_ml_training(df_featured, quick=args.quick)

        # Summary
        total_time = time.time() - start_time

        print_header("PIPELINE COMPLETE")
        print(f"✅ All steps completed successfully!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"\n📁 Outputs:")
        print(f"  - Processed data: data/processed/")
        print(f"  - Analysis reports: outputs/reports/")
        print(f"  - Trained models: outputs/models/")
        print(f"\n📊 Next steps:")
        print(f"  1. Review analysis: outputs/reports/11_questions_results.json")
        print(f"  2. Check notebook: notebooks/complete_analysis.ipynb")
        print(f"  3. Tune models for better accuracy")
        print("\n" + "=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print("   Checkpoints saved. You can resume by running again.")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
