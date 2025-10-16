import pandas as pd
import numpy as np
from pymatgen.core import Composition
from matminer.featurizers.composition import (
    ElementProperty, Stoichiometry, ElementFraction, BandCenter,
    AtomicOrbitals, ValenceOrbital, IonProperty, CohesiveEnergy,
    Miedema, YangSolidSolution, AtomicPackingEfficiency, WenAlloys,
    TMetalFraction, ElectronAffinity, ElectronegativityDiff, 
    OxidationStates, CohesiveEnergyMP
)
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

class FeatureExtractor:
    def __init__(self):
        self.featurizers = None
        self.feature_columns = None
        self.target_columns = ['Seebeck', 'electrical_conductivity', 'thermal_conductivity', 'ZT']
        self.leakage_features = ['log_electrical_conductivity', 'log_thermal_conductivity']
        
    def _safe_composition(self, formula):
        try:
            return Composition(str(formula).strip()) if formula else None
        except Exception:
            return None
    
    def _estimate_debye_temp(self, comp):
        try:
            if comp is None:
                return np.nan
            avg_mass = sum(el.atomic_mass * amt for el, amt in comp.items()) / sum(comp.values())
            return 300 * (1 + avg_mass/100)
        except:
            return np.nan
    
    def _initialize_featurizers(self):
        """Initialize comprehensive set of composition featurizers"""
        self.featurizers = [
            ElementProperty.from_preset("magpie"),
            ElementProperty.from_preset("deml"),
            Stoichiometry(),
            ElementFraction(),
            BandCenter(),
            AtomicOrbitals(),
            ValenceOrbital(),
            IonProperty(),
            CohesiveEnergy(),
            CohesiveEnergyMP(),
            Miedema(),
            WenAlloys(),
            YangSolidSolution(),
            AtomicPackingEfficiency(),
            TMetalFraction(),
            ElectronAffinity(),
            ElectronegativityDiff(),
            OxidationStates(),
        ]
        
        # Disable multiprocessing for all featurizers
        for featurizer in self.featurizers:
            if hasattr(featurizer, 'set_n_jobs'):
                featurizer.set_n_jobs(1)
    
    def _generate_comprehensive_features(self, df):
        """Step 1: Generate comprehensive features from composition"""
        print("Step 1: Generating comprehensive features...")
        
        # Create compositions
        df['composition'] = df['Formula'].apply(self._safe_composition)
        df = df.dropna(subset=['composition'])
        
        if self.featurizers is None:
            self._initialize_featurizers()
        
        feature_df = df.copy()
        
        for i, featurizer in enumerate(self.featurizers):
            try:
                print(f"  Applying {type(featurizer).__name__} ({i+1}/{len(self.featurizers)})...")
                feature_df = featurizer.featurize_dataframe(feature_df, col_id="composition", ignore_errors=True)
            except Exception as e:
                print(f"  Failed {type(featurizer).__name__}: {str(e)[:100]}...")
                continue
        
        # Add custom features
        feature_df['est_debye_temp'] = feature_df['composition'].apply(self._estimate_debye_temp)
        
        print(f"  Generated {feature_df.shape[1]} total columns")
        return feature_df
    
    def _clean_features(self, df):
        """Step 2: Clean features and remove known leakage features"""
        print("Step 2: Cleaning features...")
        
        # Get numeric feature columns (exclude metadata and targets, but include T)
        exclude_cols = ['Formula', 'composition'] + self.target_columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Remove known leakage features
        clean_feature_cols = [col for col in feature_cols if col not in self.leakage_features]
        removed_leakage = len(feature_cols) - len(clean_feature_cols)
        
        # Clean numeric data
        X = df[clean_feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"  Removed {removed_leakage} known leakage features")
        print(f"  Cleaned {X.shape[1]} features (including T)")
        
        # Combine with metadata and targets
        metadata_cols = ['Formula'] + [col for col in self.target_columns if col in df.columns]
        result_df = pd.concat([df[metadata_cols], X], axis=1)
        
        return result_df
    
    def _remove_redundant_features(self, df):
        """Step 3: Remove zero variance and highly correlated features"""
        print("Step 3: Removing redundant features...")
        
        # Separate features and targets
        feature_cols = [col for col in df.columns if col not in ['Formula'] + self.target_columns]
        X = df[feature_cols].copy()
        
        original_count = X.shape[1]
        
        # Remove zero standard deviation features
        zero_std_features = X.columns[X.std() <= 1e-10]
        if len(zero_std_features) > 0:
            X = X.drop(columns=zero_std_features)
            print(f"  Removed {len(zero_std_features)} zero variance features")
        
        # Remove highly correlated features (correlation > 0.95)
        if X.shape[1] > 1:
            corr_matrix = X.corr().abs()
            high_corr_features = set()
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_features.add(corr_matrix.columns[j])
            
            if high_corr_features:
                X = X.drop(columns=list(high_corr_features))
                print(f"  Removed {len(high_corr_features)} highly correlated features")
        
        # Store final feature columns for single sample processing
        self.feature_columns = X.columns.tolist()
        
        # Combine with metadata and targets
        metadata_cols = ['Formula'] + [col for col in self.target_columns if col in df.columns]
        result_df = pd.concat([df[metadata_cols], X], axis=1)
        
        print(f"  Final feature count: {X.shape[1]} (reduced from {original_count})")
        return result_df
    
    def _check_target_leakage(self, df):
        """Step 4: Final check for target leakage"""
        print("Step 4: Checking for target leakage...")
        
        feature_cols = [col for col in df.columns if col not in ['Formula'] + self.target_columns]
        leaked_features = []
        
        # Check for target variables in feature columns
        for target in self.target_columns:
            if target in feature_cols:
                leaked_features.append(target)
        
        # Check for variations of target names
        for feature in feature_cols:
            feature_lower = feature.lower()
            for target in self.target_columns:
                target_lower = target.lower()
                if target_lower in feature_lower and feature not in leaked_features:
                    leaked_features.append(feature)
        
        if leaked_features:
            print(f"  WARNING: Found {len(leaked_features)} potential leakage features:")
            for feature in leaked_features:
                print(f"    - {feature}")
            
            # Remove leaked features
            clean_feature_cols = [col for col in feature_cols if col not in leaked_features]
            metadata_cols = ['Formula'] + [col for col in self.target_columns if col in df.columns]
            df = df[metadata_cols + clean_feature_cols]
            
            # Update stored feature columns
            self.feature_columns = clean_feature_cols
        else:
            print("  No target leakage detected")
        
        return df
    
    def fit_transform(self, df, save_path=None):
        """Complete feature extraction pipeline for training data"""
        print("Starting feature extraction pipeline...")
        print(f"Input data shape: {df.shape}")
        
        # Step 1: Generate comprehensive features
        df = self._generate_comprehensive_features(df)
        
        # Step 2: Clean features
        df = self._clean_features(df)
        
        # Step 3: Remove redundant features
        df = self._remove_redundant_features(df)
        
        # Step 4: Check target leakage
        df = self._check_target_leakage(df)
        
        print(f"\nFeature extraction complete!")
        print(f"Final dataset shape: {df.shape}")
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Saved to: {save_path}")
        
        return df
    
    def transform_single(self, formula, temperature=300):
        """Transform a single formula for prediction"""
        if self.feature_columns is None:
            raise ValueError("FeatureExtractor must be fitted first using fit_transform()")
        
        # Create single row dataframe with temperature
        single_df = pd.DataFrame({'Formula': [formula], 'T': [temperature]})
        
        # Generate features (steps 1-2 only, no redundancy removal for single sample)
        single_df = self._generate_comprehensive_features(single_df)
        single_df = self._clean_features(single_df)
        
        # Select only the features that were kept during training
        feature_cols = [col for col in self.feature_columns if col in single_df.columns]
        missing_cols = [col for col in self.feature_columns if col not in single_df.columns]
        
        if missing_cols:
            print(f"Warning: {len(missing_cols)} features missing for single sample")
        
        # Create feature vector with same columns as training
        X_single = pd.DataFrame(columns=self.feature_columns)
        for col in self.feature_columns:
            if col in single_df.columns:
                X_single[col] = [single_df[col].iloc[0]]
            else:
                X_single[col] = [0]  # Fill missing features with 0
        
        return X_single
    
    def save_extractor(self, path):
        """Save the fitted extractor for later use"""
        extractor_data = {
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'leakage_features': self.leakage_features
        }
        with open(path, 'wb') as f:
            pickle.dump(extractor_data, f)
        print(f"Feature extractor saved to: {path}")
    
    def load_extractor(self, path):
        """Load a fitted extractor"""
        with open(path, 'rb') as f:
            extractor_data = pickle.load(f)
        
        self.feature_columns = extractor_data['feature_columns']
        self.target_columns = extractor_data['target_columns']
        self.leakage_features = extractor_data['leakage_features']
        print(f"Feature extractor loaded from: {path}")

def main():
    """Main execution for processing the ZT dataset"""
    # Load original dataset
    print("Loading ZT.csv...")
    data = pd.read_csv('Datasets/ZT.csv')
    data.columns = ['Formula', 'T', 'Seebeck', 'electrical_conductivity', 'thermal_conductivity', 'ZT']
    
    # Basic preprocessing
    for col in ['T', 'Seebeck', 'electrical_conductivity', 'thermal_conductivity', 'ZT']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.dropna()
    print(f"Clean data: {data.shape[0]} rows")
    
    # Extract features
    extractor = FeatureExtractor()
    final_df = extractor.fit_transform(data, 'processed_features.csv')
    
    # Save the fitted extractor for later use
    extractor.save_extractor('feature_extractor.pkl')
    
    print("\nFeature extraction pipeline completed successfully!")
    return final_df

if __name__ == '__main__':
    main()