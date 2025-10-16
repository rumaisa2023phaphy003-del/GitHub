#!/usr/bin/env python3
"""
Example usage of the Thermoelectric Materials Training Pipeline
Demonstrates both complete pipeline execution and individual component usage
"""

import os
import pandas as pd
from training_pipeline import ThermoelectricPipeline
from feature_extraction import FeatureExtractor
import joblib

def example_complete_pipeline():
    """Example 1: Run the complete pipeline"""
    print("="*60)
    print("EXAMPLE 1: Complete Pipeline Execution")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ThermoelectricPipeline(
        data_path='Datasets/ZT.csv',
        output_dir='example_output'
    )
    
    # Run complete pipeline
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n✅ Pipeline completed successfully!")
        print("Check 'example_output/' directory for results")
    else:
        print("\n❌ Pipeline failed. Check logs for details.")

def example_individual_components():
    """Example 2: Use individual components"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Individual Component Usage")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    data = pd.read_csv('Datasets/ZT.csv')
    data.columns = ['Formula', 'T', 'Seebeck', 'electrical_conductivity', 'thermal_conductivity', 'ZT']
    
    # Convert to numeric and clean
    for col in ['T', 'Seebeck', 'electrical_conductivity', 'thermal_conductivity', 'ZT']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    
    print(f"Loaded {len(data)} clean samples")
    
    # Extract features
    print("\n2. Extracting features...")
    extractor = FeatureExtractor()
    processed_data = extractor.fit_transform(data, 'example_features.csv')
    
    print(f"Generated {processed_data.shape[1]} columns")
    
    # Save extractor for later use
    extractor.save_extractor('example_extractor.pkl')
    
    # Train a single model (example)
    print("\n3. Training example model...")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    
    # Prepare data for ZT prediction
    targets = ['Seebeck', 'electrical_conductivity', 'thermal_conductivity', 'ZT']
    feature_cols = [col for col in processed_data.columns if col not in targets + ['Formula']]
    
    X = processed_data[feature_cols]
    y = processed_data['ZT']
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest R² for ZT: {r2:.3f}")
    
    # Save model and scaler
    joblib.dump(model, 'example_zt_model.pkl')
    joblib.dump(scaler, 'example_zt_scaler.pkl')
    
    print("✅ Individual components example completed!")

def example_predictions():
    """Example 3: Make predictions for new compounds"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Making Predictions")
    print("="*60)
    
    # Check if we have trained models
    if not os.path.exists('example_extractor.pkl'):
        print("❌ No trained extractor found. Run example_individual_components() first.")
        return
    
    if not os.path.exists('example_zt_model.pkl'):
        print("❌ No trained model found. Run example_individual_components() first.")
        return
    
    # Load fitted extractor and model
    print("\n1. Loading trained components...")
    extractor = FeatureExtractor()
    extractor.load_extractor('example_extractor.pkl')
    
    model = joblib.load('example_zt_model.pkl')
    scaler = joblib.load('example_zt_scaler.pkl')
    
    # Define test compounds
    test_compounds = [
        ('Bi2Te3', 300),    # Classic thermoelectric
        ('PbTe', 400),      # Lead telluride
        ('SnSe', 500),      # Tin selenide
        ('CoSb3', 350),     # Skutterudite
        ('Mg2Si', 450)      # Magnesium silicide
    ]
    
    print("\n2. Making predictions...")
    predictions = []
    
    for formula, temperature in test_compounds:
        try:
            # Extract features
            X_new = extractor.transform_single(formula, temperature)
            
            # Scale features
            X_scaled = scaler.transform(X_new)
            
            # Make prediction
            zt_pred = model.predict(X_scaled)[0]
            
            predictions.append({
                'Formula': formula,
                'Temperature_K': temperature,
                'Predicted_ZT': zt_pred
            })
            
            print(f"  {formula} at {temperature}K: ZT = {zt_pred:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error predicting {formula}: {e}")
    
    # Save predictions
    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv('example_predictions.csv', index=False)
        print(f"\n✅ Predictions saved to 'example_predictions.csv'")

def example_pipeline_analysis():
    """Example 4: Analyze pipeline results"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Pipeline Results Analysis")
    print("="*60)
    
    # Check if pipeline results exist
    results_path = 'example_output/output/consolidated_results.csv'
    if not os.path.exists(results_path):
        print("❌ No pipeline results found. Run example_complete_pipeline() first.")
        return
    
    # Load and analyze results
    print("\n1. Loading pipeline results...")
    results = pd.read_csv(results_path)
    
    print(f"Total models trained: {len(results)}")
    print(f"Targets: {results['Target'].unique()}")
    print(f"Models: {results['Model'].unique()}")
    
    # Best models by target
    print("\n2. Best models by target:")
    best_models = results.loc[results.groupby('Target')['Test_R2'].idxmax()]
    
    for _, row in best_models.iterrows():
        print(f"  {row['Target']}: {row['Model']} + {row['Scaler']} (R² = {row['Test_R2']:.3f})")
    
    # Performance statistics
    print("\n3. Performance statistics:")
    print(f"  Best R²: {results['Test_R2'].max():.3f}")
    print(f"  Average R²: {results['Test_R2'].mean():.3f}")
    print(f"  Worst R²: {results['Test_R2'].min():.3f}")
    
    # Model rankings
    print("\n4. Top 5 model-scaler combinations:")
    top_5 = results.nlargest(5, 'Test_R2')[['Target', 'Model', 'Scaler', 'Test_R2']]
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"  {i}. {row['Target']} - {row['Model']} + {row['Scaler']}: R² = {row['Test_R2']:.3f}")
    
    print("\n✅ Pipeline analysis completed!")

def main():
    """Run all examples"""
    print("Thermoelectric Materials Pipeline Examples")
    print("="*60)
    
    # Example 1: Complete pipeline (takes longest)
    example_complete_pipeline()
    
    # Example 2: Individual components
    example_individual_components()
    
    # Example 3: Make predictions
    example_predictions()
    
    # Example 4: Analyze results
    example_pipeline_analysis()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
    
    print("\nGenerated files:")
    files = [
        'example_output/',
        'example_features.csv',
        'example_extractor.pkl',
        'example_zt_model.pkl',
        'example_zt_scaler.pkl',
        'example_predictions.csv'
    ]
    
    for file in files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (not created)")

if __name__ == '__main__':
    # You can run individual examples by commenting/uncommenting:
    
    # Run all examples
    main()
    
    # Or run individual examples:
    # example_complete_pipeline()
    # example_individual_components()
    # example_predictions()
    # example_pipeline_analysis()