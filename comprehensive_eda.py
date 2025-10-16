import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
import os
warnings.filterwarnings('ignore')

class ComprehensiveEDA:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        
        # Handle ZT.csv format - check actual column names
        if 's' in self.df.columns and 'k' in self.df.columns:
            # Rename columns to standard names
            self.df = self.df.rename(columns={'s': 'electrical_conductivity', 'k': 'thermal_conductivity'})
        
        self.targets = ['Seebeck', 'electrical_conductivity', 'thermal_conductivity', 'ZT']
        
        # Convert target columns to numeric
        for col in self.targets + ['T']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Remove rows with NaN values after conversion
        self.df = self.df.dropna()
        
        self.feature_cols = [col for col in self.df.columns if col not in self.targets + ['Formula']]
        os.makedirs('eda_plots', exist_ok=True)
        
    def basic_info(self):
        """Generate basic dataset information"""
        print("Dataset Overview")
        print("="*50)
        print(f"Shape: {self.df.shape}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Targets: {len(self.targets)}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(f"\nMissing values: {missing[missing > 0]}")
        else:
            print("\nNo missing values found")
            
        # Data types
        print(f"\nData types:")
        print(self.df.dtypes.value_counts())
        
    def target_distributions(self):
        """Analyze target variable distributions"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, target in enumerate(self.targets):
            # Linear distribution
            axes[i].hist(self.df[target], bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{target} Distribution')
            axes[i].set_xlabel(target)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Log distribution (if positive values)
            if self.df[target].min() > 0:
                axes[i+4].hist(np.log10(self.df[target]), bins=50, alpha=0.7, edgecolor='black', color='orange')
                axes[i+4].set_title(f'Log10({target}) Distribution')
                axes[i+4].set_xlabel(f'Log10({target})')
                axes[i+4].set_ylabel('Frequency')
                axes[i+4].grid(True, alpha=0.3)
            else:
                axes[i+4].text(0.5, 0.5, 'Contains non-positive values', 
                              ha='center', va='center', transform=axes[i+4].transAxes)
                axes[i+4].set_title(f'Log10({target}) - Not Applicable')
        
        plt.tight_layout()
        plt.savefig('eda_plots/target_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def target_statistics(self):
        """Generate target statistics table"""
        stats_data = []
        for target in self.targets:
            data = self.df[target]
            stats_data.append({
                'Target': target,
                'Count': len(data),
                'Mean': data.mean(),
                'Std': data.std(),
                'Min': data.min(),
                'Q25': data.quantile(0.25),
                'Median': data.median(),
                'Q75': data.quantile(0.75),
                'Max': data.max(),
                'Skewness': stats.skew(data),
                'Kurtosis': stats.kurtosis(data),
                'Range_Orders': np.log10(data.max() / data.min()) if data.min() > 0 else np.nan
            })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv('eda_plots/target_statistics.csv', index=False)
        
        # Visualization of statistics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Range comparison
        ax1.bar(self.targets, [stats_df[stats_df['Target']==t]['Range_Orders'].iloc[0] for t in self.targets])
        ax1.set_title('Value Range (Orders of Magnitude)')
        ax1.set_ylabel('Log10(Max/Min)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Skewness
        ax2.bar(self.targets, [stats_df[stats_df['Target']==t]['Skewness'].iloc[0] for t in self.targets])
        ax2.set_title('Distribution Skewness')
        ax2.set_ylabel('Skewness')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.tick_params(axis='x', rotation=45)
        
        # Coefficient of variation
        cv = stats_df['Std'] / stats_df['Mean']
        ax3.bar(self.targets, cv)
        ax3.set_title('Coefficient of Variation')
        ax3.set_ylabel('Std/Mean')
        ax3.tick_params(axis='x', rotation=45)
        
        # Box plot comparison
        target_data = [self.df[target] for target in self.targets]
        ax4.boxplot(target_data, labels=self.targets)
        ax4.set_title('Target Value Distributions')
        ax4.set_ylabel('Values')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('eda_plots/target_statistics_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def correlation_analysis(self):
        """Analyze correlations between targets"""
        # Target correlation matrix
        target_corr = self.df[self.targets].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(target_corr, annot=True, cmap='RdBu_r', center=0, 
                    square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
        plt.title('Target Variable Correlations')
        plt.tight_layout()
        plt.savefig('eda_plots/target_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Pairwise scatter plots
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        for i, target1 in enumerate(self.targets):
            for j, target2 in enumerate(self.targets):
                if i == j:
                    axes[i, j].hist(self.df[target1], bins=30, alpha=0.7)
                    axes[i, j].set_title(f'{target1}')
                else:
                    axes[i, j].scatter(self.df[target2], self.df[target1], alpha=0.5, s=10)
                    r, p = pearsonr(self.df[target2], self.df[target1])
                    axes[i, j].set_title(f'r={r:.3f}, p={p:.3e}')
                
                if i == 3:
                    axes[i, j].set_xlabel(target2)
                if j == 0:
                    axes[i, j].set_ylabel(target1)
        
        plt.tight_layout()
        plt.savefig('eda_plots/target_pairwise_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def feature_analysis(self):
        """Analyze feature characteristics"""
        # Feature statistics
        feature_data = self.df[self.feature_cols]
        
        # Basic statistics
        feature_stats = {
            'zero_variance': (feature_data.std() == 0).sum(),
            'near_zero_variance': (feature_data.std() < 1e-6).sum(),
            'constant_features': feature_data.nunique()[feature_data.nunique() == 1].count(),
            'high_missing': (feature_data.isnull().sum() > len(feature_data) * 0.1).sum(),
            'infinite_values': np.isinf(feature_data).sum().sum(),
            'total_features': len(self.feature_cols)
        }
        
        # Feature correlation with targets
        target_correlations = {}
        for target in self.targets:
            correlations = []
            for feature in self.feature_cols[:100]:  # Limit to first 100 features for speed
                try:
                    corr, _ = pearsonr(feature_data[feature], self.df[target])
                    correlations.append(abs(corr))
                except:
                    correlations.append(0)
            target_correlations[target] = correlations
        
        # Plot feature quality metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Feature variance distribution
        variances = feature_data.std()
        ax1.hist(variances[variances > 0], bins=50, alpha=0.7)
        ax1.set_xlabel('Standard Deviation')
        ax1.set_ylabel('Number of Features')
        ax1.set_title('Feature Variance Distribution')
        ax1.set_yscale('log')
        
        # Feature correlation with targets (top 100 features)
        corr_df = pd.DataFrame(target_correlations, index=self.feature_cols[:100])
        ax2.boxplot([corr_df[target] for target in self.targets], labels=self.targets)
        ax2.set_title('Feature-Target Correlations (Top 100 Features)')
        ax2.set_ylabel('Absolute Correlation')
        ax2.tick_params(axis='x', rotation=45)
        
        # Feature quality summary
        quality_labels = list(feature_stats.keys())[:-1]
        quality_values = list(feature_stats.values())[:-1]
        ax3.bar(quality_labels, quality_values)
        ax3.set_title('Feature Quality Issues')
        ax3.set_ylabel('Number of Features')
        ax3.tick_params(axis='x', rotation=45)
        
        # Top correlated features for each target
        top_features = {}
        for target in self.targets:
            top_idx = np.argsort(target_correlations[target])[-10:]
            top_features[target] = [self.feature_cols[i] for i in top_idx]
        
        # Display top features as text
        ax4.axis('off')
        text_content = "Top 5 Features by Target:\n\n"
        for target in self.targets:
            text_content += f"{target}:\n"
            for i, feature in enumerate(top_features[target][-5:]):
                corr_val = target_correlations[target][self.feature_cols.index(feature)]
                text_content += f"  {i+1}. {feature[:20]}... (r={corr_val:.3f})\n"
            text_content += "\n"
        
        ax4.text(0.05, 0.95, text_content, transform=ax4.transAxes, 
                fontsize=8, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('eda_plots/feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save feature statistics
        with open('eda_plots/feature_statistics.txt', 'w') as f:
            f.write("Feature Analysis Summary\n")
            f.write("="*30 + "\n\n")
            for key, value in feature_stats.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nTop correlated features by target:\n")
            for target in self.targets:
                f.write(f"\n{target}:\n")
                for i, feature in enumerate(top_features[target][-10:]):
                    corr_val = target_correlations[target][self.feature_cols.index(feature)]
                    f.write(f"  {i+1:2d}. {feature} (r={corr_val:.4f})\n")
    
    def outlier_analysis(self):
        """Detect and visualize outliers"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        outlier_summary = {}
        
        for i, target in enumerate(self.targets):
            data = self.df[target]
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_iqr = ((data < lower_bound) | (data > upper_bound)).sum()
            
            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            outliers_zscore = (z_scores > 3).sum()
            
            outlier_summary[target] = {
                'IQR_outliers': outliers_iqr,
                'ZScore_outliers': outliers_zscore,
                'Percentage_IQR': (outliers_iqr / len(data)) * 100,
                'Percentage_ZScore': (outliers_zscore / len(data)) * 100
            }
            
            # Box plot with outliers
            axes[i].boxplot(data, vert=True)
            axes[i].set_title(f'{target} Outliers\nIQR: {outliers_iqr} ({outlier_summary[target]["Percentage_IQR"]:.1f}%)')
            axes[i].set_ylabel(target)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('eda_plots/outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save outlier summary
        outlier_df = pd.DataFrame(outlier_summary).T
        outlier_df.to_csv('eda_plots/outlier_summary.csv')
        
    def generate_summary_report(self):
        """Generate comprehensive EDA summary report"""
        with open('eda_plots/EDA_SUMMARY_REPORT.md', 'w') as f:
            f.write(f"""# Comprehensive EDA Report

## Dataset Overview
- **Shape**: {self.df.shape}
- **Features**: {len(self.feature_cols)}
- **Targets**: {len(self.targets)}
- **Memory Usage**: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

## Target Variables
{', '.join(self.targets)}

## Key Findings

### Data Quality
- Missing values: {'None' if self.df.isnull().sum().sum() == 0 else 'Present'}
- Infinite values: {'Present' if np.isinf(self.df.select_dtypes(include=[np.number])).sum().sum() > 0 else 'None'}

### Target Characteristics
""")
            
            for target in self.targets:
                data = self.df[target]
                f.write(f"""
#### {target}
- Range: {data.min():.3e} to {data.max():.3e}
- Mean ± Std: {data.mean():.3e} ± {data.std():.3e}
- Skewness: {stats.skew(data):.3f}
- Orders of magnitude: {np.log10(data.max() / data.min()) if data.min() > 0 else 'N/A'}
""")
            
            f.write(f"""
## Files Generated
- `target_distributions.png` - Distribution plots for all targets
- `target_statistics.csv` - Detailed target statistics
- `target_correlations.png` - Target correlation heatmap
- `target_pairwise_plots.png` - Pairwise scatter plots
- `feature_analysis.png` - Feature quality analysis
- `outlier_analysis.png` - Outlier detection plots
- `feature_statistics.txt` - Feature analysis summary

## Recommendations
1. Consider log transformation for targets with wide ranges
2. Check feature importance for model selection
3. Handle outliers based on domain knowledge
4. Monitor high-correlation target pairs for multicollinearity
""")
    
    def run_complete_eda(self):
        """Run all EDA analyses"""
        print("Starting Comprehensive EDA...")
        
        self.basic_info()
        print("\n1. Analyzing target distributions...")
        self.target_distributions()
        
        print("2. Computing target statistics...")
        self.target_statistics()
        
        print("3. Analyzing correlations...")
        self.correlation_analysis()
        
        print("4. Analyzing features...")
        self.feature_analysis()
        
        print("5. Detecting outliers...")
        self.outlier_analysis()
        
        print("6. Generating summary report...")
        self.generate_summary_report()
        
        print(f"\nEDA Complete! All plots saved to 'eda_plots/' directory")

def main():
    """Main EDA execution"""
    # Try different possible data files
    data_files = [
        'final_cleaned_comprehensive_features.csv',
        'cleaned_comprehensive_features.csv',
        'comprehensive_features.csv',
        'Datasets/ZT.csv'
    ]
    
    data_file = None
    for file in data_files:
        if os.path.exists(file):
            data_file = file
            break
    
    if data_file is None:
        print("No data file found. Please ensure one of these files exists:")
        for file in data_files:
            print(f"  - {file}")
        return
    
    print(f"Using data file: {data_file}")
    
    # Run EDA
    eda = ComprehensiveEDA(data_file)
    eda.run_complete_eda()

if __name__ == '__main__':
    main()