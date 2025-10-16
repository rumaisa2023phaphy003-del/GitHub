"""
Example usage of the consolidated feature extraction pipeline
"""
from feature_extraction import FeatureExtractor
import pandas as pd

def example_training_pipeline():
    """Example: Process training data"""
    print("=== TRAINING PIPELINE EXAMPLE ===")
    
    # Load your training data
    data = pd.read_csv('Datasets/ZT.csv')
    data.columns = ['Formula', 'T', 'Seebeck', 'electrical_conductivity', 'thermal_conductivity', 'ZT']
    
    # Basic preprocessing
    for col in ['T', 'Seebeck', 'electrical_conductivity', 'thermal_conductivity', 'ZT']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    
    # Extract features using the pipeline
    extractor = FeatureExtractor()
    processed_data = extractor.fit_transform(data, 'training_features.csv')
    
    # Save the fitted extractor for later use with single samples
    extractor.save_extractor('trained_feature_extractor.pkl')
    
    print(f"Training features extracted: {processed_data.shape}")
    return processed_data, extractor

def example_single_prediction():
    """Example: Process single sample for prediction"""
    print("\n=== SINGLE SAMPLE PREDICTION EXAMPLE ===")
    
    # Load the fitted extractor
    extractor = FeatureExtractor()
    extractor.load_extractor('trained_feature_extractor.pkl')
    
    # Process a single formula with temperature
    test_formula = "Bi2Te3"
    test_temperature = 300  # Kelvin
    features = extractor.transform_single(test_formula, test_temperature)
    
    print(f"Single sample features extracted: {features.shape}")
    print(f"Feature vector ready for model prediction")
    
    return features

if __name__ == '__main__':
    # Run training pipeline
    processed_data, extractor = example_training_pipeline()
    
    # Run single sample prediction
    single_features = example_single_prediction()