#!/usr/bin/env python3
import os
import pandas as pd
from datetime import datetime

def generate_summary(output_dir='pipeline_output', data_path='Datasets/ZT.csv'):
    """Generate only the pipeline summary"""
    
    # Load data to get shapes
    raw_data = None
    processed_data = None
    
    if os.path.exists(data_path):
        raw_data = pd.read_csv(data_path)
    
    processed_path = f"{output_dir}/processed_features.csv"
    if os.path.exists(processed_path):
        processed_data = pd.read_csv(processed_path)
    
    summary_path = f"{output_dir}/PIPELINE_SUMMARY.md"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"""# Thermoelectric Materials Training Pipeline Summary

## Pipeline Execution
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Input Data**: {data_path}
- **Output Directory**: {output_dir}

## Data Processing Summary
- **Raw Data Shape**: {raw_data.shape if raw_data is not None else 'N/A'}
- **Processed Data Shape**: {processed_data.shape if processed_data is not None else 'N/A'}
- **Features Generated**: {processed_data.shape[1] - 5 if processed_data is not None else 'N/A'} (excluding Formula + 4 targets)

## Pipeline Steps Completed
1. ✅ Data Loading and Cleaning
2. ✅ Comprehensive Exploratory Data Analysis
3. ✅ Feature Extraction (Composition-based features)
4. ✅ Model Training (13 algorithms × 3 scalers × 4 targets)
5. ✅ Summary Generation

## Output Files Structure
```
{output_dir}/
├── cleaned_raw_data.csv              # Cleaned input data
├── processed_features.csv            # Features + targets
├── feature_extractor.pkl             # Fitted feature extractor
├── pipeline.log                      # Execution log
├── eda_plots/                        # EDA visualizations
│   ├── target_distributions.png
│   ├── target_correlations.png
│   ├── feature_analysis.png
│   └── EDA_SUMMARY_REPORT.md
├── log_space/                        # Log-space models
│   ├── electrical_conductivity/
│   └── thermal_conductivity/
├── linear_space/                     # Linear-space models
│   ├── Seebeck/
│   └── ZT/
└── output/                           # Training results
    ├── consolidated_results.csv
    ├── best_models.csv
    └── CONSOLIDATED_TRAINING_REPORT.md
```

## Key Features
- **Composition Featurization**: 18 different featurizers from matminer
- **Automated Space Selection**: Log vs linear space based on data characteristics
- **Comprehensive Model Comparison**: 13 algorithms with 3 scaling methods
- **Outlier Handling**: IQR-based outlier removal per target
- **Cross-validation**: 5-fold CV for model stability assessment

## Usage for Predictions
```python
from feature_extraction import FeatureExtractor
import joblib

# Load fitted extractor
extractor = FeatureExtractor()
extractor.load_extractor('{output_dir}/feature_extractor.pkl')

# Extract features for new compound
X_new = extractor.transform_single('Bi2Te3', temperature=300)

# Load best model for target (example: ZT)
model = joblib.load('{output_dir}/linear_space/ZT/best_model.pkl')
scaler = joblib.load('{output_dir}/linear_space/ZT/best_scaler.pkl')

# Make prediction
X_scaled = scaler.transform(X_new)
prediction = model.predict(X_scaled)
```

## Next Steps
1. Review EDA findings in `eda_plots/EDA_SUMMARY_REPORT.md`
2. Check best models in `output/best_models.csv`
3. Use fitted models for new predictions
""")
    
    print(f"Pipeline summary generated: {summary_path}")

if __name__ == '__main__':
    generate_summary()