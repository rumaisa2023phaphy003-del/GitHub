#!/usr/bin/env python3
"""
Comprehensive Training Pipeline for Thermoelectric Materials
Integrates EDA, Feature Extraction, and Model Training
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from comprehensive_eda import ComprehensiveEDA
from feature_extraction import FeatureExtractor
from consolidated_training import ConsolidatedTrainer

class ThermoelectricPipeline:
    def __init__(self, data_path='Datasets/ZT.csv', output_dir='pipeline_output'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.raw_data = None
        self.processed_data = None
        self.feature_extractor = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def log_message(self, message, level="INFO"):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
        # Also save to log file
        with open(f"{self.output_dir}/pipeline.log", "a") as f:
            f.write(f"[{timestamp}] {level}: {message}\n")
    
    def step_1_load_data(self):
        """Step 1: Load and validate raw data"""
        self.log_message("Step 1: Loading raw data")
        
        try:
            self.raw_data = pd.read_csv(self.data_path)
            
            # Standardize column names for ZT.csv
            if 's' in self.raw_data.columns and 'k' in self.raw_data.columns:
                self.raw_data.columns = ['Formula', 'T', 'Seebeck', 'electrical_conductivity', 'thermal_conductivity', 'ZT']
            
            # Convert numeric columns
            numeric_cols = ['T', 'Seebeck', 'electrical_conductivity', 'thermal_conductivity', 'ZT']
            for col in numeric_cols:
                if col in self.raw_data.columns:
                    self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')
            
            # Remove rows with missing values
            initial_rows = len(self.raw_data)
            self.raw_data = self.raw_data.dropna()
            final_rows = len(self.raw_data)
            
            self.log_message(f"Loaded {initial_rows} rows, {final_rows} after cleaning")
            self.log_message(f"Dataset shape: {self.raw_data.shape}")
            
            # Save cleaned raw data
            self.raw_data.to_csv(f"{self.output_dir}/cleaned_raw_data.csv", index=False)
            
            return True
            
        except Exception as e:
            self.log_message(f"Error loading data: {e}", "ERROR")
            return False
    
    def step_2_exploratory_analysis(self):
        """Step 2: Comprehensive EDA"""
        self.log_message("Step 2: Running comprehensive EDA")
        
        try:
            # Save data temporarily for EDA
            temp_data_path = f"{self.output_dir}/temp_eda_data.csv"
            self.raw_data.to_csv(temp_data_path, index=False)
            
            # Run EDA
            eda = ComprehensiveEDA(temp_data_path)
            eda.run_complete_eda()
            
            # Move EDA outputs to our pipeline directory
            if os.path.exists('eda_plots'):
                import shutil
                if os.path.exists(f"{self.output_dir}/eda_plots"):
                    shutil.rmtree(f"{self.output_dir}/eda_plots")
                shutil.move('eda_plots', f"{self.output_dir}/eda_plots")
            
            # Clean up temp file
            os.remove(temp_data_path)
            
            self.log_message("EDA completed successfully")
            return True
            
        except Exception as e:
            self.log_message(f"Error in EDA: {e}", "ERROR")
            return False
    
    def step_3_feature_extraction(self):
        """Step 3: Extract comprehensive features"""
        self.log_message("Step 3: Extracting features")
        
        try:
            # Initialize feature extractor
            self.feature_extractor = FeatureExtractor()
            
            # Extract features
            self.processed_data = self.feature_extractor.fit_transform(
                self.raw_data, 
                save_path=f"{self.output_dir}/processed_features.csv"
            )
            
            # Save feature extractor for later use
            self.feature_extractor.save_extractor(f"{self.output_dir}/feature_extractor.pkl")
            
            self.log_message(f"Feature extraction completed: {self.processed_data.shape}")
            return True
            
        except Exception as e:
            self.log_message(f"Error in feature extraction: {e}", "ERROR")
            return False
    
    def step_4_model_training(self):
        """Step 4: Train comprehensive models"""
        self.log_message("Step 4: Training models")
        
        try:
            # Save processed data in output directory
            data_path = os.path.join(self.output_dir, 'final_cleaned_comprehensive_features.csv')
            self.processed_data.to_csv(data_path, index=False)
            
            # Initialize trainer with output directory
            trainer = ConsolidatedTrainer(output_base=self.output_dir)
            trainer._create_output_dirs()
            
            # Prepare data
            targets = ['Seebeck', 'electrical_conductivity', 'thermal_conductivity', 'ZT']
            feature_cols = [col for col in self.processed_data.columns if col not in targets + ['Formula']]
            X = self.processed_data[feature_cols]
            
            self.log_message(f"Training with {len(feature_cols)} features on {len(targets)} targets")
            
            # Train all targets
            for target in targets:
                if target in self.processed_data.columns:
                    y = self.processed_data[target]
                    trainer.train_target(X, y, target)
            
            # Generate summary
            trainer.generate_summary_report()
            
            self.log_message("Model training completed successfully")
            return True
            
        except Exception as e:
            self.log_message(f"Error in model training: {e}", "ERROR")
            return False
    
    def step_5_generate_summary(self):
        """Step 5: Generate pipeline summary"""
        self.log_message("Step 5: Generating pipeline summary")
        
        try:
            summary_path = f"{self.output_dir}/PIPELINE_SUMMARY.md"
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"""# Thermoelectric Materials Training Pipeline Summary

## Pipeline Execution
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Input Data**: {self.data_path}
- **Output Directory**: {self.output_dir}

## Data Processing Summary
- **Raw Data Shape**: {self.raw_data.shape if self.raw_data is not None else 'N/A'}
- **Processed Data Shape**: {self.processed_data.shape if self.processed_data is not None else 'N/A'}
- **Features Generated**: {self.processed_data.shape[1] - 5 if self.processed_data is not None else 'N/A'} (excluding Formula + 4 targets)

## Pipeline Steps Completed
1. ✅ Data Loading and Cleaning
2. ✅ Comprehensive Exploratory Data Analysis
3. ✅ Feature Extraction (Composition-based features)
4. ✅ Model Training (13 algorithms × 3 scalers × 4 targets)
5. ✅ Summary Generation

## Output Files Structure
```
{self.output_dir}/
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
extractor.load_extractor('{self.output_dir}/feature_extractor.pkl')

# Extract features for new compound
X_new = extractor.transform_single('Bi2Te3', temperature=300)

# Load best model for target (example: ZT)
model = joblib.load('{self.output_dir}/linear_space/ZT/best_model.pkl')
scaler = joblib.load('{self.output_dir}/linear_space/ZT/best_scaler.pkl')

# Make prediction
X_scaled = scaler.transform(X_new)
prediction = model.predict(X_scaled)
```

## Next Steps
1. Review EDA findings in `eda_plots/EDA_SUMMARY_REPORT.md`
2. Check best models in `output/best_models.csv`
3. Use fitted models for new predictions
""")
            
            self.log_message(f"Pipeline summary saved to {summary_path}")
            return True
            
        except Exception as e:
            self.log_message(f"Error generating summary: {e}", "ERROR")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        self.log_message("Starting Thermoelectric Materials Training Pipeline")
        self.log_message("="*60)
        
        steps = [
            ("Data Loading", self.step_1_load_data),
            ("Exploratory Data Analysis", self.step_2_exploratory_analysis),
            ("Feature Extraction", self.step_3_feature_extraction),
            ("Model Training", self.step_4_model_training),
            ("Summary Generation", self.step_5_generate_summary)
        ]
        
        for step_name, step_func in steps:
            self.log_message(f"Starting: {step_name}")
            success = step_func()
            
            if not success:
                self.log_message(f"Pipeline failed at: {step_name}", "ERROR")
                return False
            
            self.log_message(f"Completed: {step_name}")
        
        self.log_message("="*60)
        self.log_message("Pipeline completed successfully!")
        self.log_message(f"All outputs saved to: {self.output_dir}")
        
        return True

def main():
    """Main execution with command line arguments"""
    parser = argparse.ArgumentParser(description='Thermoelectric Materials Training Pipeline')
    parser.add_argument('--data', default='Datasets/ZT.csv', help='Path to input data file')
    parser.add_argument('--output', default='pipeline_output', help='Output directory')
    parser.add_argument('--skip-eda', action='store_true', help='Skip EDA step')
    parser.add_argument('--skip-features', action='store_true', help='Skip feature extraction step')
    parser.add_argument('--skip-training', action='store_true', help='Skip training step')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ThermoelectricPipeline(data_path=args.data, output_dir=args.output)
    
    # Run pipeline with optional step skipping
    skip_count = sum([args.skip_eda, args.skip_features, args.skip_training])
    if skip_count >= 3:
        print("Error: Cannot skip all steps (EDA, features, and training)")
        return
    
    # Custom execution based on arguments
    if args.skip_eda or args.skip_features or args.skip_training:
        pipeline.log_message("Running partial pipeline")
        
        # Always load data
        if not pipeline.step_1_load_data():
            return
        
        # Skip EDA if requested
        if not args.skip_eda:
            if not pipeline.step_2_exploratory_analysis():
                return
        
        # Skip feature extraction if requested
        if not args.skip_features:
            if not pipeline.step_3_feature_extraction():
                return
        else:
            # If skipping features, try to load existing processed data
            processed_path = f"{pipeline.output_dir}/processed_features.csv"
            if os.path.exists(processed_path):
                pipeline.log_message("Loading existing processed features")
                pipeline.processed_data = pd.read_csv(processed_path)
                pipeline.log_message(f"Loaded processed data: {pipeline.processed_data.shape}")
            else:
                pipeline.log_message("No existing processed features found. Cannot skip feature extraction.", "ERROR")
                return
        
        # Skip training if requested
        if not args.skip_training:
            if not pipeline.step_4_model_training():
                return
        
        # Always generate summary
        pipeline.step_5_generate_summary()
    else:
        # Run complete pipeline
        pipeline.run_complete_pipeline()

if __name__ == '__main__':
    main()