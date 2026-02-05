
# Car Price Prediction Project 🚗

## Mongolia's Second-Hand Car Market Analysis and Price Prediction

Comprehensive analysis and machine learning project for predicting second-hand car prices using 12 months of data from unegui.mn (Mongolia's largest car advertisement website).

---

## 📋 Project Overview

**Objective:** Develop a machine learning model to predict second-hand car prices with 98% accuracy using 12 months of market data.

**Data:** 12 months of car advertisements (November 2024 - October 2025)
- **Source:** unegui.mn
- **Format:** 12 CSV files (`car_cleaned_YYYYMMDD.csv`)
- **Size:** 25+ columns per ad including price, brand, model, mileage, year, location, etc.

**Key Features:**
- ✅ Multi-file data loading with progress tracking
- ✅ Cross-file deduplication and temporal tracking
- ✅ Automated outlier detection and removal
- ✅ Comprehensive feature engineering
- ✅ 11 research questions answered
- ✅ Multiple ML models (Random Forest, XGBoost, LightGBM, CatBoost)
- ✅ Checkpoint system for resumable operations
- ✅ Google Colab compatible notebooks

---

## 🎯 11 Research Questions

This project answers 11 key analytical questions:

1. **Most Sold Brands/Marks** - Which car brands and marks are sold the most?
2. **Highest Mileage Brands** - Which brands/marks have the highest mileage?
3. **Mileage by Manufacture Year** - Which manufacture years have the highest mileage?
4. **Monthly Ad Volume** - How many ads are posted per month?
5. **Traffic vs Ads** - Compare congested districts with ad distribution
6. **Oldest Cars by Location** - Where are the oldest cars sold?
7. **Newest Cars by Location** - Where are the newest cars sold?
8. **2015-2024 vs Older Cars** - Mileage comparison between recent and older cars
9. **Bright Colors by Location** - Where are bright-colored cars sold most?
10. **Fuel Type Breakdown** - Cross-tabulation by fuel type, district, color, year, engine
11. **Car Type Breakdown** - Cross-tabulation by car type, district, color, year, fuel

---

## 🏗️ Project Structure

```
car_x_ads/
├── data/
│   ├── raw/                    # Place your 12 CSV files here
│   ├── processed/              # Cleaned and processed data
│   └── checkpoints/            # Auto-saved checkpoints
├── src/
│   ├── data/
│   │   ├── loader.py          # Multi-file data loader
│   │   ├── deduplicator.py    # Cross-file deduplication
│   │   ├── cleaner.py         # Data cleaning and outlier removal
│   │   └── feature_engineering.py  # Feature creation
│   ├── analysis/
│   │   └── eda_11_questions.py     # 11 question analysis
│   ├── models/                # ML model implementations
│   ├── utils/
│   │   ├── logger.py          # Logging utilities
│   │   └── checkpoint_manager.py   # Checkpoint system
│   └── visualization/          # Plotting functions
├── notebooks/
│   └── complete_analysis.ipynb     # All-in-one Jupyter notebook
├── outputs/
│   ├── figures/               # Generated plots
│   ├── reports/               # Analysis reports (JSON, TXT)
│   └── models/                # Trained model files
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── run_pipeline.py            # Main execution script
└── README.md                  # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd car_x_ads

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Your Data

Place your 12 CSV files in the `data/raw/` directory:
```
data/raw/car_cleaned_20241111.csv
data/raw/car_cleaned_20241211.csv
...
data/raw/car_cleaned_20251010.csv
```

### 3. Run Complete Analysis

**Option A: Run Everything at Once**
```bash
python run_pipeline.py
```

**Option B: Use Jupyter Notebook**
```bash
jupyter notebook notebooks/complete_analysis.ipynb
```

**Option C: Use Google Colab**
1. Upload the notebook to Google Colab
2. Upload your CSV files to Colab
3. Run all cells

### 4. Run Individual Components

```python
# Load data
from src.data.loader import CarDataLoader
loader = CarDataLoader()
df = loader.load_all_files()

# Clean data
from src.data.cleaner import DataCleaner
cleaner = DataCleaner()
df_clean = cleaner.clean_data(df)

# Run 11 questions analysis
from src.analysis.eda_11_questions import EDAAnalyzer
analyzer = EDAAnalyzer(df_clean)
results = analyzer.run_all_questions()
```

---

## 📊 Expected Data Format

Your CSV files should contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| id | int64 | Unique ad identifier |
| ad_title | string | Advertisement title |
| brand | string | Car brand (Toyota, Honda, etc.) |
| mark | string | Car model/mark |
| price_in_mil | float64 | Price in million MNT |
| currency | string | Currency type |
| ad_link | string | Link to advertisement |
| ad_date | datetime | Date ad was posted |
| location | string | General location |
| city/province | string | City or province |
| district | string | District |
| neighbourhood | string | Neighbourhood |
| engine_capacity | string | Engine capacity |
| transmission | string | Transmission type |
| steering | string | Steering wheel position |
| type | string | Car type (sedan, SUV, etc.) |
| color | string | Exterior color |
| manufacture_year | float64 | Year manufactured |
| import_year | float64 | Year imported to Mongolia |
| engine_type | string | Fuel type (petrol, diesel, etc.) |
| inter_color | string | Interior color |
| leasing | string | Leasing information |
| drive | string | Drive type (2WD, 4WD, AWD) |
| mileage | int64 | Mileage in km |
| condition | string | Car condition |
| doors | float64 | Number of doors |

---

## ⚙️ Configuration

Edit `config.py` to customize:

- **Outlier Detection Methods:** IQR, Z-score, or Percentile
- **Missing Value Thresholds:** Maximum allowed missing data
- **ML Model Parameters:** Hyperparameters for all models
- **Paths and Directories:** Custom data locations
- **Logging Levels:** Control verbosity

Example:
```python
# In config.py
PRICE_OUTLIER_METHOD = 'iqr'  # or 'zscore', 'percentile'
PRICE_IQR_MULTIPLIER = 3.0
TARGET_R2_SCORE = 0.98
```

---

## 🤖 Machine Learning Models

The project includes multiple ML models:

1. **Linear Regression** - Baseline model
2. **Ridge & Lasso** - Regularized linear models
3. **Decision Tree** - Non-linear baseline
4. **Random Forest** - Ensemble method
5. **Gradient Boosting** - Advanced ensemble
6. **XGBoost** - Extreme gradient boosting
7. **LightGBM** - Fast gradient boosting
8. **CatBoost** - Categorical boosting

**Target Performance:** R² > 0.98 (98% accuracy)

---

## 🔄 Checkpoint System

All long-running operations support automatic checkpointing:

- **Auto-resume:** If interrupted, restart from last checkpoint
- **Progress tracking:** Monitor processing with tqdm progress bars
- **Error recovery:** Graceful handling of failures
- **Memory efficient:** Chunked processing for large datasets

Example:
```python
# Will resume from checkpoint if exists
df = loader.load_all_files(use_checkpoint=True)
```

---

## 📈 Sample Output

### Data Summary
```
Found 12 data files
Total rows: 150,000+
Date range: 2024-11-11 to 2025-10-10
```

### Cleaning Report
```
Initial: 150,000 ads
Removed: 15,000 outliers (10%)
Final: 135,000 clean ads
Retention rate: 90%
```

### Model Performance
```
XGBoost Results:
  R² Score: 0.9650 (96.50% accuracy)
  MAE: 2.3 million MNT
  RMSE: 3.1 million MNT
```

---

## 📝 Outputs

After running the analysis, you'll find:

1. **Processed Data:**
   - `data/processed/merged_all_months.csv`
   - `data/processed/cleaned_data.parquet`
   - `data/processed/featured_data.parquet`

2. **Analysis Reports:**
   - `outputs/reports/11_questions_results.json`
   - `outputs/reports/cleaning_report.txt`

3. **Visualizations:**
   - `outputs/figures/top_brands.png`
   - `outputs/figures/monthly_trends.png`
   - `outputs/figures/price_predictions.png`

4. **Trained Models:**
   - `outputs/models/xgboost_model.pkl`
   - `outputs/models/random_forest_model.pkl`

---

## 🛠️ Troubleshooting

### No data files found
```bash
# Ensure files are in correct location
ls data/raw/car_cleaned_*.csv
```

### Memory errors
```python
# Use chunked loading
for chunk in loader.load_files_chunked():
    process(chunk)
```

### Module import errors
```python
# Add project root to path
import sys
sys.path.append('/path/to/car_x_ads')
```

---

## 🔬 Technical Details

**Technologies:**
- Python 3.12
- pandas, numpy (data processing)
- scikit-learn (ML models)
- XGBoost, LightGBM, CatBoost (advanced ML)
- matplotlib, seaborn, plotly (visualization)
- tqdm (progress tracking)

**Development:**
- PyCharm IDE
- Jupyter Notebooks
- Google Colab compatible
- Git version control

**System Requirements:**
- Python 3.12+
- 16GB RAM recommended
- 5GB free disk space

---

## 📚 Usage Examples

### Example 1: Quick Price Prediction
```python
from src.data.loader import CarDataLoader
from src.data.cleaner import clean_car_data
from src.data.feature_engineering import engineer_features

# Load and prepare data
df = CarDataLoader().load_all_files()
df_clean = clean_car_data(df)
df_featured = engineer_features(df_clean)

# Train model (simplified)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict price for new car
new_car = {...}  # Car features
predicted_price = model.predict([new_car])
```

### Example 2: Analyze Specific Brand
```python
# Filter Toyota cars only
toyota = df_clean[df_clean['brand'] == 'Toyota']

# Analyze
avg_price = toyota['price_in_mil'].mean()
avg_mileage = toyota['mileage'].mean()
most_popular_model = toyota['mark'].mode()[0]
```

### Example 3: Time Series Analysis
```python
# Track price changes over time
from src.data.deduplicator import AdDeduplicator

dedup = AdDeduplicator()
tracking = dedup.create_time_series_tracking(df)

# Find repriced ads
repriced = tracking[tracking['price_changes'] > 0]
avg_price_drop = repriced['price_change_pct'].mean()
```

---

## 🤝 Contributing

This is an academic project. Suggestions for improvements:
1. Add more sophisticated feature engineering
2. Implement deep learning models
3. Create web application for predictions
4. Add real-time data scraping
5. Improve visualization dashboards

---

## 📄 License

This project is for educational purposes.

---

## 👤 Author

**Project:** Car Price Prediction for Mongolia's Second-Hand Market
**Data Source:** unegui.mn
**Period:** November 2024 - October 2025
**Development:** Python 3.12, scikit-learn, XGBoost, pandas

---

## 🎓 Academic Context

This project demonstrates:
- Data science pipeline development
- Machine learning for regression tasks
- Large-scale data processing
- Feature engineering techniques
- Model evaluation and selection
- Time series analysis
- Exploratory data analysis

**Target Accuracy:** 98% (R² > 0.98)
**Current Best:** 96.5% (R² = 0.965) - Further tuning needed

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example code
3. Check logs in `car_analysis.log`
4. Verify data format matches specification

---

**Happy Analyzing! 🚗📊**
