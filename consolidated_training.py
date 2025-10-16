import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ConsolidatedTrainer:
    def __init__(self, output_base='.'):
        self.models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0),
            'DecisionTree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100),
            'ExtraTrees': ExtraTreesRegressor(random_state=42, n_estimators=100),
            'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBRegressor(random_state=42, n_estimators=100),
            'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1, n_estimators=100),
            'SVR': SVR(kernel='rbf'),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'NeuralNetwork': MLPRegressor(random_state=42, max_iter=500, hidden_layer_sizes=(100, 50))
        }
        
        self.scalers = {
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler(),
            'MinMaxScaler': MinMaxScaler()
        }
        
        # Define which targets benefit from log transformation
        self.log_targets = ['electrical_conductivity', 'thermal_conductivity']
        self.results = []
        self.output_base = output_base
        
    def _create_output_dirs(self):
        """Create output directory structure"""
        dirs = ['output', 'output/models', 'output/plots', 'output/predictions', 
                'output/metrics', 'output/summaries', 'log_space', 'linear_space']
        for target in ['Seebeck', 'electrical_conductivity', 'thermal_conductivity', 'ZT']:
            dirs.extend([f'log_space/{target}', f'linear_space/{target}'])
        
        for d in dirs:
            full_path = os.path.join(self.output_base, d)
            os.makedirs(full_path, exist_ok=True)
    
    def _should_use_log_space(self, target, y_values):
        """Determine if log space is beneficial for this target"""
        if target in self.log_targets:
            # Check if values span multiple orders of magnitude
            if y_values.min() > 0:
                ratio = y_values.max() / y_values.min()
                return ratio > 100  # Use log if range > 2 orders of magnitude
        return False
    
    def _prepare_target_data(self, y, target, use_log):
        """Prepare target data for training"""
        if use_log:
            # Add small constant to avoid log(0)
            y_log = np.log10(y + 1e-6)
            return y_log, y  # Return both log and linear versions
        return y, y
    
    def _calculate_metrics(self, y_true, y_pred, y_true_linear=None, y_pred_linear=None, use_log=False):
        """Calculate comprehensive metrics"""
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred)
        }
        
        if use_log and y_true_linear is not None and y_pred_linear is not None:
            # MAPE in linear space
            mape = np.mean(np.abs((y_true_linear - y_pred_linear) / y_true_linear)) * 100
            metrics['mape_linear'] = mape
            metrics['r2_linear'] = r2_score(y_true_linear, y_pred_linear)
        
        return metrics
    
    def _save_model_artifacts(self, model, scaler, normalizer, target, model_name, scaler_name, space_type):
        """Save model and preprocessing artifacts"""
        base_path = os.path.join(self.output_base, space_type, target)
        
        # Save model and preprocessors
        joblib.dump(model, os.path.join(base_path, f'{model_name}_{scaler_name}_model.pkl'))
        joblib.dump(scaler, os.path.join(base_path, f'{model_name}_{scaler_name}_scaler.pkl'))
        if normalizer:
            joblib.dump(normalizer, os.path.join(base_path, f'{model_name}_{scaler_name}_normalizer.pkl'))
    
    def _create_visualizations(self, results_df, target, space_type):
        """Create performance visualization plots"""
        if results_df.empty:
            return
            
        # Performance comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        top_10 = results_df.head(10)
        model_labels = [f"{row['Model']}\n({row['Scaler']})" for _, row in top_10.iterrows()]
        
        # R² Score
        ax1.barh(range(len(top_10)), top_10['Test_R2'], color='skyblue', alpha=0.8)
        ax1.set_yticks(range(len(top_10)))
        ax1.set_yticklabels(model_labels, fontsize=8)
        ax1.set_xlabel('R² Score')
        ax1.set_title(f'Test R² - {target} ({space_type})')
        ax1.grid(axis='x', alpha=0.3)
        
        # RMSE
        ax2.barh(range(len(top_10)), top_10['Test_RMSE'], color='lightcoral', alpha=0.8)
        ax2.set_yticks(range(len(top_10)))
        ax2.set_yticklabels(model_labels, fontsize=8)
        ax2.set_xlabel('RMSE')
        ax2.set_title(f'Test RMSE - {target} ({space_type})')
        ax2.grid(axis='x', alpha=0.3)
        
        # MAE
        ax3.barh(range(len(top_10)), top_10['Test_MAE'], color='lightgreen', alpha=0.8)
        ax3.set_yticks(range(len(top_10)))
        ax3.set_yticklabels(model_labels, fontsize=8)
        ax3.set_xlabel('MAE')
        ax3.set_title(f'Test MAE - {target} ({space_type})')
        ax3.grid(axis='x', alpha=0.3)
        
        # Cross-validation stability
        ax4.barh(range(len(top_10)), top_10['CV_R2_Std'], color='gold', alpha=0.8)
        ax4.set_yticks(range(len(top_10)))
        ax4.set_yticklabels(model_labels, fontsize=8)
        ax4.set_xlabel('CV R² Std')
        ax4.set_title(f'CV Stability - {target} ({space_type})')
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_base, space_type, target, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Model-scaler heatmap
        models_list = results_df['Model'].unique()
        scalers_list = results_df['Scaler'].unique()
        
        heatmap_data = np.zeros((len(models_list), len(scalers_list)))
        for i, model in enumerate(models_list):
            for j, scaler in enumerate(scalers_list):
                r2_score = results_df[(results_df['Model'] == model) & (results_df['Scaler'] == scaler)]['Test_R2'].values
                if len(r2_score) > 0:
                    heatmap_data[i, j] = r2_score[0]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                    xticklabels=scalers_list, yticklabels=models_list,
                    cbar_kws={'label': 'R² Score'})
        plt.title(f'Model Performance Heatmap - {target} ({space_type})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_base, space_type, target, 'model_scaler_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _remove_outliers(self, X, y, target):
        """Remove outliers using IQR method for specific target"""
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outlier_mask = (y < lower_bound) | (y > upper_bound)
        outliers_count = outlier_mask.sum()
        
        # Remove outliers
        X_clean = X[~outlier_mask]
        y_clean = y[~outlier_mask]
        
        print(f"  Removed {outliers_count} outliers ({outliers_count/len(y)*100:.1f}%) for {target}")
        print(f"  Clean dataset: {len(y_clean)} samples")
        
        return X_clean, y_clean
    
    def train_target(self, X, y, target):
        """Train all models for a specific target"""
        print(f"\n{'='*60}")
        print(f"Training target: {target}")
        print('='*60)
        
        # Remove target-specific outliers
        X_clean, y_clean = self._remove_outliers(X, y, target)
        
        # Determine if log space should be used
        use_log = self._should_use_log_space(target, y_clean)
        space_type = 'log_space' if use_log else 'linear_space'
        
        print(f"Using {'log' if use_log else 'linear'} space for {target}")
        
        # Prepare target data
        y_train_space, y_linear = self._prepare_target_data(y_clean, target, use_log)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_train_space, test_size=0.1, random_state=42)
        if use_log:
            _, _, y_train_linear, y_test_linear = train_test_split(X_clean, y_linear, test_size=0.1, random_state=42)
        else:
            y_train_linear, y_test_linear = y_train, y_test
        
        target_results = []
        
        for scaler_name, scaler in self.scalers.items():
            print(f"\nUsing {scaler_name}:")
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Apply normalization for comprehensive training
            normalizer = Normalizer(norm='l2')
            X_train_processed = normalizer.fit_transform(X_train_scaled)
            X_test_processed = normalizer.transform(X_test_scaled)
            
            for model_name, model in self.models.items():
                try:
                    print(f"  Training {model_name}...")
                    
                    # Train model
                    model.fit(X_train_processed, y_train)
                    
                    # Predictions
                    y_train_pred = model.predict(X_train_processed)
                    y_test_pred = model.predict(X_test_processed)
                    
                    # Convert back to linear space if needed
                    if use_log:
                        y_train_pred_linear = 10**y_train_pred - 1e-6
                        y_test_pred_linear = 10**y_test_pred - 1e-6
                    else:
                        y_train_pred_linear = y_train_pred
                        y_test_pred_linear = y_test_pred
                    
                    # Calculate metrics
                    train_metrics = self._calculate_metrics(
                        y_train, y_train_pred, y_train_linear, y_train_pred_linear, use_log
                    )
                    test_metrics = self._calculate_metrics(
                        y_test, y_test_pred, y_test_linear, y_test_pred_linear, use_log
                    )
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_processed, y_train, cv=5, scoring='r2')
                    
                    # Store results
                    result = {
                        'Target': target,
                        'Model': model_name,
                        'Scaler': scaler_name,
                        'Space': space_type,
                        'Train_R2': train_metrics['r2'],
                        'Test_R2': test_metrics['r2'],
                        'Train_RMSE': train_metrics['rmse'],
                        'Test_RMSE': test_metrics['rmse'],
                        'Train_MAE': train_metrics['mae'],
                        'Test_MAE': test_metrics['mae'],
                        'CV_R2_Mean': cv_scores.mean(),
                        'CV_R2_Std': cv_scores.std(),
                        'Overfitting': train_metrics['r2'] - test_metrics['r2']
                    }
                    
                    if use_log:
                        result['Test_MAPE_Linear'] = test_metrics.get('mape_linear', np.nan)
                        result['Test_R2_Linear'] = test_metrics.get('r2_linear', np.nan)
                    
                    target_results.append(result)
                    self.results.append(result)
                    
                    # Save model artifacts
                    self._save_model_artifacts(model, scaler, normalizer, target, model_name, scaler_name, space_type)
                    
                    # Save predictions
                    pred_df = pd.DataFrame({
                        'Actual': y_test_linear,
                        'Predicted': y_test_pred_linear,
                        'Residuals': y_test_linear - y_test_pred_linear
                    })
                    pred_df.to_csv(os.path.join(self.output_base, space_type, target, f'{model_name}_{scaler_name}_predictions.csv'), index=False)
                    
                    print(f"    R² = {test_metrics['r2']:.4f}, RMSE = {test_metrics['rmse']:.4f}")
                    
                except Exception as e:
                    print(f"    Error with {model_name}: {e}")
        
        # Create visualizations and save results for this target
        if target_results:
            target_df = pd.DataFrame(target_results).sort_values('Test_R2', ascending=False)
            target_df.to_csv(os.path.join(self.output_base, space_type, target, 'results.csv'), index=False)
            self._create_visualizations(target_df, target, space_type)
            
            # Print best model for this target
            best = target_df.iloc[0]
            print(f"\nBest model for {target}: {best['Model']} + {best['Scaler']}")
            print(f"Test R²: {best['Test_R2']:.3f}, RMSE: {best['Test_RMSE']:.3f}")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if not self.results:
            return
            
        results_df = pd.DataFrame(self.results)
        
        # Overall performance heatmap
        pivot_df = results_df.pivot_table(values='Test_R2', index='Model', columns='Target', aggfunc='max')
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0.3, fmt='.3f')
        plt.title('Best R² Score by Model and Target')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_base, 'output', 'overall_performance_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Best models by target
        best_models = results_df.loc[results_df.groupby('Target')['Test_R2'].idxmax()]
        
        # Save comprehensive results
        results_df.to_csv(os.path.join(self.output_base, 'output', 'consolidated_results.csv'), index=False)
        best_models.to_csv(os.path.join(self.output_base, 'output', 'best_models.csv'), index=False)
        
        # Generate summary report
        with open(os.path.join(self.output_base, 'output', 'CONSOLIDATED_TRAINING_REPORT.md'), 'w') as f:
            f.write(f"""# Consolidated Training Report

## Training Summary
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Total Models Trained**: {len(self.results)}
- **Targets**: {len(results_df['Target'].unique())}
- **Best Overall R²**: {results_df['Test_R2'].max():.4f}

## Best Models by Target
""")
            
            for _, row in best_models.iterrows():
                space_info = f" ({row['Space']})" if 'Space' in row else ""
                f.write(f"""
### {row['Target']}{space_info}
- **Model**: {row['Model']} + {row['Scaler']}
- **Test R²**: {row['Test_R2']:.4f}
- **Test RMSE**: {row['Test_RMSE']:.4f}
- **CV Score**: {row['CV_R2_Mean']:.4f} ± {row['CV_R2_Std']:.4f}
""")
        
        print(f"\nConsolidated training complete!")
        print(f"Best models by target:")
        for _, row in best_models.iterrows():
            print(f"  {row['Target']}: {row['Model']} + {row['Scaler']} (R² = {row['Test_R2']:.4f})")

def main():
    """Main training pipeline"""
    print("Consolidated Training Pipeline")
    print("="*50)
    
    # Initialize trainer
    trainer = ConsolidatedTrainer()
    trainer._create_output_dirs()
    
    # Load data
    try:
        df = pd.read_csv('final_cleaned_comprehensive_features.csv')
        print(f"Loaded data: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Prepare features and targets
    targets = ['Seebeck', 'electrical_conductivity', 'thermal_conductivity', 'ZT']
    feature_cols = [col for col in df.columns if col not in targets]
    X = df[feature_cols]
    
    print(f"Features: {len(feature_cols)}")
    print(f"Targets: {len(targets)}")
    
    # Train all targets
    for target in targets:
        if target in df.columns:
            y = df[target]
            trainer.train_target(X, y, target)
    
    # Generate summary
    trainer.generate_summary_report()

if __name__ == '__main__':
    main()