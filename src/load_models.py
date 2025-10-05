import joblib
import pandas as pd

def load_and_inspect_models():
    """Load the trained models and show their information"""
    
    print("Loading trained models from .pkl files...")
    
    # Load the models
    rf_model = joblib.load('results/random_forest.pkl')
    xgb_model = joblib.load('results/xgboost.pkl')
    
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    
    # Random Forest info
    print(f"\nRandom Forest Model:")
    print(f"  - Number of trees: {rf_model.n_estimators}")
    print(f"  - Max depth: {rf_model.max_depth}")
    print(f"  - Features used: {rf_model.n_features_in_}")
    print(f"  - Feature names: {list(rf_model.feature_names_in_)}")
    
    # XGBoost info
    print(f"\nXGBoost Model:")
    print(f"  - Number of trees: {xgb_model.n_estimators}")
    print(f"  - Max depth: {xgb_model.max_depth}")
    print(f"  - Learning rate: {xgb_model.learning_rate}")
    print(f"  - Features used: {xgb_model.n_features_in_}")
    print(f"  - Feature names: {list(xgb_model.feature_names_in_)}")
    
    print(f"\nBoth models are ready for prediction!")
    print(f"Use: python src/predict.py --url 'your-url-here'")

if __name__ == "__main__":
    load_and_inspect_models()

