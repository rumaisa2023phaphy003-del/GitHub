# Thermoelectric Materials Machine Learning Pipeline

A comprehensive machine learning pipeline for predicting thermoelectric properties from chemical composition. This repository provides end-to-end functionality for exploratory data analysis, feature extraction, and model training for thermoelectric materials research.

## 🎯 Overview

This pipeline predicts four key thermoelectric properties:
- **Seebeck coefficient** (μV/K)
- **Electrical conductivity** (S/cm)
- **Thermal conductivity** (W/mK)
- **Figure of merit (ZT)** (dimensionless)

## 📊 Dataset

The pipeline works with the ZT.csv dataset containing:
- **Formula**: Chemical composition (e.g., "Bi2Te3")
- **T**: Temperature (K)
- **Seebeck**: Seebeck coefficient
- **s**: Electrical conductivity (renamed to electrical_conductivity)
- **k**: Thermal conductivity (renamed to thermal_conductivity)
- **ZT**: Thermoelectric figure of merit

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
python training_pipeline.py
```

This will:
1. Load and clean the data
2. Perform comprehensive EDA
3. Extract composition-based features
4. Train 13 different ML models
5. Generate comprehensive reports

### Custom Execution

```bash
# Skip EDA (faster execution)
python training_pipeline.py --skip-eda

# Skip feature extraction (use existing processed features)
python training_pipeline.py --skip-features

# Skip training (EDA and feature extraction only)
python training_pipeline.py --skip-training

# Skip multiple steps (but not all)
python training_pipeline.py --skip-eda --skip-features

# Custom data file and output directory
python training_pipeline.py --data "my_data.csv" --output "my_results"
```

## 📁 Repository Structure

```
├── Datasets/
│   └── ZT.csv                          # Raw thermoelectric data
├── comprehensive_eda.py                # Exploratory data analysis
├── feature_extraction.py               # Composition featurization
├── consolidated_training.py            # Model training
├── training_pipeline.py                # Complete pipeline
├── requirements.txt                    # Dependencies
├── README.md                          # This file
└── pipeline_output/                   # Generated outputs
    ├── eda_plots/                     # EDA visualizations
    ├── log_space/                     # Log-space models
    ├── linear_space/                  # Linear-space models
    └── output/                        # Training results
```

## 🔧 Individual Components

### 1. Exploratory Data Analysis (`comprehensive_eda.py`)

Comprehensive analysis including:
- Target variable distributions
- Correlation analysis
- Feature quality assessment
- Outlier detection
- Statistical summaries

```python
from comprehensive_eda import ComprehensiveEDA

eda = ComprehensiveEDA('Datasets/ZT.csv')
eda.run_complete_eda()
```

**Outputs:**
- Distribution plots for all targets
- Correlation heatmaps
- Feature analysis plots
- Outlier detection visualizations
- Comprehensive EDA report

### 2. Feature Extraction (`feature_extraction.py`)

Extracts 1000+ composition-based features using matminer:

**Featurizers included:**
- ElementProperty (Magpie + DEML presets)
- Stoichiometry
- ElementFraction
- BandCenter
- AtomicOrbitals
- ValenceOrbital
- IonProperty
- CohesiveEnergy
- Miedema alloy features
- Yang solid solution features
- And more...

```python
from feature_extraction import FeatureExtractor

extractor = FeatureExtractor()
processed_data = extractor.fit_transform(raw_data)
extractor.save_extractor('feature_extractor.pkl')
```

**Features:**
- Automatic leakage detection and removal
- Zero variance feature removal
- High correlation feature removal (>0.95)
- Single sample transformation for predictions

### 3. Model Training (`consolidated_training.py`)

Trains 13 different algorithms with comprehensive evaluation:

**Algorithms:**
- Linear models: LinearRegression, Ridge, Lasso, ElasticNet
- Tree-based: DecisionTree, RandomForest, ExtraTrees, GradientBoosting
- Boosting: XGBoost, LightGBM
- Other: SVR, KNN, Neural Network

**Features:**
- Automatic log/linear space selection
- 3 different scaling methods
- Outlier removal per target
- Cross-validation
- Comprehensive metrics

```python
from consolidated_training import ConsolidatedTrainer

trainer = ConsolidatedTrainer()
trainer.train_target(X, y, 'ZT')
trainer.generate_summary_report()
```

### 4. Complete Pipeline (`training_pipeline.py`)

Orchestrates all components with:
- Comprehensive logging
- Error handling
- Progress tracking
- Automated output organization

## 📈 Model Performance

The pipeline automatically:
- Selects optimal space (log vs linear) per target
- Removes outliers using IQR method
- Applies appropriate scaling
- Evaluates using multiple metrics (R², RMSE, MAE, MAPE)
- Performs 5-fold cross-validation

**Typical performance ranges:**
- **ZT**: R² = 0.7-0.9
- **Seebeck**: R² = 0.6-0.8
- **Electrical conductivity**: R² = 0.5-0.7 (log space)
- **Thermal conductivity**: R² = 0.4-0.6 (log space)

## 🔮 Making Predictions

### For New Compounds

```python
from feature_extraction import FeatureExtractor
import joblib

# Load fitted extractor
extractor = FeatureExtractor()
extractor.load_extractor('pipeline_output/feature_extractor.pkl')

# Extract features for new compound
X_new = extractor.transform_single('Bi2Te3', temperature=300)

# Load best model (example for ZT)
model = joblib.load('pipeline_output/linear_space/ZT/best_model.pkl')
scaler = joblib.load('pipeline_output/linear_space/ZT/best_scaler.pkl')

# Make prediction
X_scaled = scaler.transform(X_new)
zt_prediction = model.predict(X_scaled)[0]

print(f"Predicted ZT for Bi2Te3 at 300K: {zt_prediction:.3f}")
```

### Batch Predictions

```python
# For multiple compounds
compounds = ['Bi2Te3', 'PbTe', 'SnSe']
temperatures = [300, 400, 500]

predictions = []
for formula in compounds:
    for temp in temperatures:
        X_new = extractor.transform_single(formula, temp)
        X_scaled = scaler.transform(X_new)
        pred = model.predict(X_scaled)[0]
        predictions.append({
            'Formula': formula,
            'Temperature': temp,
            'Predicted_ZT': pred
        })

results_df = pd.DataFrame(predictions)
```

## 📊 Output Files

### EDA Outputs (`eda_plots/`)
- `target_distributions.png` - Distribution analysis
- `target_correlations.png` - Correlation heatmap
- `feature_analysis.png` - Feature quality analysis
- `outlier_analysis.png` - Outlier detection
- `EDA_SUMMARY_REPORT.md` - Comprehensive EDA report

### Training Outputs (`output/`)
- `consolidated_results.csv` - All model results
- `best_models.csv` - Best model per target
- `overall_performance_heatmap.png` - Performance comparison
- `CONSOLIDATED_TRAINING_REPORT.md` - Training summary

### Model Files (`log_space/` & `linear_space/`)
For each target and model combination:
- `{model}_{scaler}_model.pkl` - Trained model
- `{model}_{scaler}_scaler.pkl` - Fitted scaler
- `{model}_{scaler}_predictions.csv` - Test predictions
- `performance_comparison.png` - Performance plots
- `model_scaler_heatmap.png` - Model-scaler heatmap

## 🛠️ Customization

### Adding New Featurizers

```python
# In feature_extraction.py, add to _initialize_featurizers():
from matminer.featurizers.composition import NewFeaturizer

self.featurizers.append(NewFeaturizer())
```

### Adding New Models

```python
# In consolidated_training.py, add to __init__():
from sklearn.ensemble import NewModel

self.models['NewModel'] = NewModel(parameters)
```

### Custom Target Processing

```python
# Override _should_use_log_space() in consolidated_training.py
def _should_use_log_space(self, target, y_values):
    if target == 'my_target':
        return True  # Force log space
    return super()._should_use_log_space(target, y_values)
```

## 🔬 Research Applications

This pipeline is designed for:
- **Materials discovery**: Screen new thermoelectric compounds
- **Property prediction**: Estimate performance before synthesis
- **Optimization**: Identify promising composition ranges
- **Data analysis**: Understand structure-property relationships

## 📚 Dependencies

Key packages:
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Machine learning
- `xgboost`, `lightgbm` - Gradient boosting
- `matplotlib`, `seaborn` - Visualization
- `pymatgen` - Materials analysis
- `matminer` - Materials featurization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📄 License

This project is open source. Please cite if used in research.

## 🔗 References

- [Matminer](https://hackingmaterials.lbl.gov/matminer/) - Feature extraction
- [Pymatgen](https://pymatgen.org/) - Materials analysis
- [Scikit-learn](https://scikit-learn.org/) - Machine learning

## 📞 Support

For questions or issues:
1. Check the generated log files in `pipeline_output/`
2. Review the EDA and training reports
3. Open an issue on GitHub

---

**Happy materials discovery! 🧪⚡**