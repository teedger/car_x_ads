"""
EDA Analysis - Answers to 11 Research Questions
Comprehensive analysis of car advertisement data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import REPORTS_DIR, TOP_N_BRANDS, TOP_N_MARKS, TOP_N_DISTRICTS, STANDARD_COLORS
from src.utils.logger import get_logger


logger = get_logger(__name__)


class EDAAnalyzer:
    """Performs exploratory data analysis answering 11 key questions"""

    def __init__(self, df):
        self.df = df
        self.logger = logger
        self.results = {}

    def question_1_most_sold_brands(self):
        """Q1: Which car brands and marks are sold the most?"""
        self.logger.info("Q1: Analyzing most sold brands/marks...")

        # Top brands
        brand_counts = self.df['brand'].value_counts().head(TOP_N_BRANDS)

        # Top marks
        mark_counts = self.df['mark'].value_counts().head(TOP_N_MARKS)

        # Brand-mark combinations
        brand_mark = self.df.groupby(['brand', 'mark']).size().sort_values(ascending=False).head(20)

        self.results['q1'] = {
            'top_brands': brand_counts.to_dict(),
            'top_marks': mark_counts.to_dict(),
            'top_combinations': brand_mark.to_dict()
        }

        return self.results['q1']

    def question_2_highest_mileage_brands(self):
        """Q2: Which brands/marks have the highest mileage?"""
        self.logger.info("Q2: Analyzing highest mileage brands/marks...")

        # Average mileage by brand
        brand_mileage = self.df.groupby('brand')['mileage'].agg(['mean', 'median', 'max']).sort_values('mean', ascending=False).head(TOP_N_BRANDS)

        # Average mileage by mark
        mark_mileage = self.df.groupby('mark')['mileage'].agg(['mean', 'median', 'max']).sort_values('mean', ascending=False).head(TOP_N_MARKS)

        self.results['q2'] = {
            'brand_mileage': brand_mileage.to_dict(),
            'mark_mileage': mark_mileage.to_dict()
        }

        return self.results['q2']

    def question_3_mileage_by_manufacture_year(self):
        """Q3: Which manufacture years have highest mileage?"""
        self.logger.info("Q3: Analyzing mileage by manufacture year...")

        year_mileage = self.df.groupby('manufacture_year')['mileage'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)

        self.results['q3'] = {
            'year_mileage': year_mileage.to_dict(),
            'highest_avg_mileage_year': int(year_mileage['mean'].idxmax()),
            'highest_avg_mileage': float(year_mileage['mean'].max())
        }

        return self.results['q3']

    def question_4_monthly_ad_volume(self):
        """Q4: How many ads are posted per month?"""
        self.logger.info("Q4: Analyzing monthly ad volume...")

        if 'collection_date' in self.df.columns:
            monthly_counts = self.df.groupby(pd.to_datetime(self.df['collection_date']).dt.to_period('M')).size()
            monthly_counts.index = monthly_counts.index.astype(str)

            self.results['q4'] = {
                'monthly_counts': monthly_counts.to_dict(),
                'avg_monthly_ads': float(monthly_counts.mean()),
                'max_month': str(monthly_counts.idxmax()),
                'min_month': str(monthly_counts.idxmin())
            }
        else:
            self.results['q4'] = {'error': 'No collection_date column'}

        return self.results['q4']

    def question_5_traffic_vs_ads(self):
        """Q5: Compare traffic congestion districts with ad data"""
        self.logger.info("Q5: Analyzing ads by district (traffic correlation requires external data)...")

        district_counts = self.df['district'].value_counts().head(TOP_N_DISTRICTS)

        self.results['q5'] = {
            'ads_by_district': district_counts.to_dict(),
            'note': 'Traffic congestion data needed for full analysis'
        }

        return self.results['q5']

    def question_6_oldest_cars_by_location(self):
        """Q6: Where are the oldest cars sold?"""
        self.logger.info("Q6: Analyzing oldest cars by location...")

        if 'car_age' in self.df.columns:
            age_col = 'car_age'
        else:
            self.df['temp_age'] = 2025 - self.df['manufacture_year']
            age_col = 'temp_age'

        district_age = self.df.groupby('district')[age_col].agg(['mean', 'median']).sort_values('mean', ascending=False).head(TOP_N_DISTRICTS)

        self.results['q6'] = {
            'districts_with_oldest_cars': district_age.to_dict()
        }

        return self.results['q6']

    def question_7_newest_cars_by_location(self):
        """Q7: Where are the newest cars sold?"""
        self.logger.info("Q7: Analyzing newest cars by location...")

        if 'car_age' in self.df.columns:
            age_col = 'car_age'
        else:
            age_col = 'temp_age'

        district_age = self.df.groupby('district')[age_col].agg(['mean', 'median']).sort_values('mean').head(TOP_N_DISTRICTS)

        self.results['q7'] = {
            'districts_with_newest_cars': district_age.to_dict()
        }

        return self.results['q7']

    def question_8_mileage_comparison_2015_2024(self):
        """Q8: Compare mileage of 2015-2024 vs older cars"""
        self.logger.info("Q8: Comparing mileage 2015-2024 vs older...")

        recent = self.df[self.df['manufacture_year'].between(2015, 2024)]
        older = self.df[self.df['manufacture_year'] < 2015]

        self.results['q8'] = {
            'recent_2015_2024': {
                'count': len(recent),
                'mean_mileage': float(recent['mileage'].mean()),
                'median_mileage': float(recent['mileage'].median())
            },
            'older_pre_2015': {
                'count': len(older),
                'mean_mileage': float(older['mileage'].mean()),
                'median_mileage': float(older['mileage'].median())
            }
        }

        return self.results['q8']

    def question_9_bright_colors_by_location(self):
        """Q9: Where are bright colored cars sold most?"""
        self.logger.info("Q9: Analyzing bright colored cars by location...")

        # Identify bright colors (non-standard)
        self.df['is_bright'] = ~self.df['color'].str.lower().isin([c.lower() for c in STANDARD_COLORS])

        bright_by_district = self.df[self.df['is_bright']].groupby('district').size().sort_values(ascending=False).head(TOP_N_DISTRICTS)

        self.results['q9'] = {
            'bright_cars_by_district': bright_by_district.to_dict(),
            'total_bright_cars': int(self.df['is_bright'].sum()),
            'bright_car_percentage': float((self.df['is_bright'].sum() / len(self.df)) * 100)
        }

        return self.results['q9']

    def question_10_fuel_type_breakdown(self):
        """Q10: Breakdown by fuel type, district, color, year, engine capacity"""
        self.logger.info("Q10: Creating fuel type breakdowns...")

        results = {}

        # By fuel type
        if 'engine_type' in self.df.columns:
            results['by_fuel_type'] = self.df['engine_type'].value_counts().to_dict()

            # Fuel type by district
            fuel_district = pd.crosstab(self.df['engine_type'], self.df['district'])
            results['fuel_by_district'] = fuel_district.to_dict()

            # Fuel type by color
            if 'color' in self.df.columns:
                fuel_color = pd.crosstab(self.df['engine_type'], self.df['color'])
                results['fuel_by_color_sample'] = fuel_color.iloc[:5, :5].to_dict()

            # Fuel type by year
            if 'manufacture_year' in self.df.columns:
                fuel_year = self.df.groupby(['engine_type', 'manufacture_year']).size().unstack(fill_value=0)
                results['fuel_by_year_shape'] = fuel_year.shape

        self.results['q10'] = results
        return self.results['q10']

    def question_11_car_type_breakdown(self):
        """Q11: Breakdown by car type, district, color, year, fuel"""
        self.logger.info("Q11: Creating car type breakdowns...")

        results = {}

        # By car type
        if 'type' in self.df.columns:
            results['by_car_type'] = self.df['type'].value_counts().to_dict()

            # Type by district
            type_district = pd.crosstab(self.df['type'], self.df['district'])
            results['type_by_district'] = type_district.to_dict()

            # Type by color
            if 'color' in self.df.columns:
                type_color = pd.crosstab(self.df['type'], self.df['color'])
                results['type_by_color_sample'] = type_color.iloc[:5, :5].to_dict()

            # Type by fuel
            if 'engine_type' in self.df.columns:
                type_fuel = pd.crosstab(self.df['type'], self.df['engine_type'])
                results['type_by_fuel'] = type_fuel.to_dict()

        self.results['q11'] = results
        return self.results['q11']

    def run_all_questions(self, save_results=True):
        """Run all 11 questions"""
        self.logger.info("Running all 11 EDA questions...")

        for i in range(1, 12):
            method = getattr(self, f'question_{i}_{self.__get_question_name(i)}')
            method()

        if save_results:
            self.save_results()

        return self.results

    def __get_question_name(self, q_num):
        """Get question method name"""
        names = {
            1: 'most_sold_brands',
            2: 'highest_mileage_brands',
            3: 'mileage_by_manufacture_year',
            4: 'monthly_ad_volume',
            5: 'traffic_vs_ads',
            6: 'oldest_cars_by_location',
            7: 'newest_cars_by_location',
            8: 'mileage_comparison_2015_2024',
            9: 'bright_colors_by_location',
            10: 'fuel_type_breakdown',
            11: 'car_type_breakdown'
        }
        return names[q_num]

    def save_results(self):
        """Save results to file"""
        import json

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_file = REPORTS_DIR / '11_questions_results.json'

        # Convert numpy types to native Python
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif pd.isna(obj):
                return None
            return obj

        results_clean = convert_types(self.results)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_clean, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    from src.utils.logger import setup_logger
    from src.data.loader import CarDataLoader
    from src.data.cleaner import DataCleaner
    from src.data.feature_engineering import FeatureEngineer

    setup_logger("eda", level='INFO')

    # Load and prepare data
    loader = CarDataLoader()
    df = loader.load_all_files()

    if df is not None:
        cleaner = DataCleaner()
        df_clean = cleaner.clean_data(df)

        fe = FeatureEngineer()
        df_featured = fe.engineer_features(df_clean)

        # Run analysis
        analyzer = EDAAnalyzer(df_featured)
        results = analyzer.run_all_questions()

        print("\n" + "=" * 70)
        print("11 QUESTIONS ANALYSIS COMPLETE")
        print("=" * 70)
        print("\nResults saved to outputs/reports/11_questions_results.json")
