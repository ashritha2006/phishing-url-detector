import joblib
import pandas as pd
import numpy as np

def demo_pkl_usage():
    """Demonstrate how PKL files work with a simple example"""
    
    print("="*60)
    print("PKL FILES DEMONSTRATION")
    print("="*60)
    
    print("\n1. CREATING A SIMPLE MODEL TO DEMONSTRATE:")
    
    # Create some sample data
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 0, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    
    print("   Sample data created:")
    print(df)
    
    # Train a simple model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    X = df[['feature1', 'feature2']]
    y = df['target']
    model.fit(X, y)
    
    print(f"\n   Model trained: {type(model)}")
    print(f"   Model parameters: {model.coef_}")
    
    print("\n2. SAVING MODEL TO PKL FILE:")
    joblib.dump(model, 'demo_model.pkl')
    print("   Model saved to: demo_model.pkl")
    
    print("\n3. LOADING MODEL FROM PKL FILE:")
    loaded_model = joblib.load('demo_model.pkl')
    print(f"   Model loaded: {type(loaded_model)}")
    print(f"   Same parameters: {np.array_equal(model.coef_, loaded_model.coef_)}")
    
    print("\n4. USING LOADED MODEL:")
    test_data = [[2.5, 25]]
    prediction = loaded_model.predict(test_data)
    probability = loaded_model.predict_proba(test_data)
    
    print(f"   Test data: {test_data[0]}")
    print(f"   Prediction: {prediction[0]}")
    print(f"   Probability: {probability[0]}")
    
    print("\n5. YOUR PHISHING DETECTION MODELS:")
    print("   - random_forest.pkl: 179.98 MB (200 trees)")
    print("   - xgboost.pkl: 0.27 MB (gradient boosting)")
    print("   - Both contain trained models ready for prediction")
    
    print("\n6. HOW TO USE YOUR PKL FILES:")
    print("   python src/predict.py --url 'https://example.com'")
    
    # Clean up demo file
    import os
    if os.path.exists('demo_model.pkl'):
        os.remove('demo_model.pkl')
        print("\n   Demo file cleaned up.")
    
    print("\n" + "="*60)
    print("PKL FILES = SAVED PYTHON OBJECTS")
    print("Think of them as 'frozen' Python objects that can be 'thawed' later!")
    print("="*60)

if __name__ == "__main__":
    demo_pkl_usage()

