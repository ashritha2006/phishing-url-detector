import pandas as pd
import numpy as np
import joblib
import argparse
from urllib.parse import urlparse
import re

def extract_features(url):
    """Extract features from a URL for prediction - matching training data format"""
    features = {}
    
    # Basic URL features (matching training data exactly)
    features['url_length'] = float(len(url))
    features['num_dots'] = float(url.count('.'))
    features['has_ip'] = 1.0 if re.match(r'^\d+\.\d+\.\d+\.\d+', urlparse(url).netloc) else 0.0
    features['has_at'] = 1.0 if '@' in url else 0.0
    features['prefix_suffix'] = 1.0 if url.startswith(('http://', 'https://')) else 0.0
    features['subdomain_count'] = float(len([x for x in urlparse(url).netloc.split('.') if x]))
    features['https_in_domain'] = 1.0 if 'https' in urlparse(url).netloc.lower() else 0.0
    
    # Entropy calculation
    url_chars = [c for c in url if c.isalnum()]
    if len(url_chars) > 0:
        char_counts = {}
        for char in url_chars:
            char_counts[char] = char_counts.get(char, 0) + 1
        entropy = -sum((count/len(url_chars)) * np.log2(count/len(url_chars)) 
                      for count in char_counts.values())
        features['entropy'] = float(entropy)
    else:
        features['entropy'] = 0.0
    
    # Keyword features (matching training data)
    features['kw_login'] = 1.0 if 'login' in url.lower() else 0.0
    features['kw_secure'] = 1.0 if 'secure' in url.lower() else 0.0
    features['kw_update'] = 1.0 if 'update' in url.lower() else 0.0
    features['kw_verify'] = 1.0 if 'verify' in url.lower() else 0.0
    features['kw_bank'] = 1.0 if 'bank' in url.lower() else 0.0
    features['kw_account'] = 1.0 if 'account' in url.lower() else 0.0
    features['kw_signin'] = 1.0 if 'signin' in url.lower() else 0.0
    features['kw_password'] = 1.0 if 'password' in url.lower() else 0.0
    
    return features

def main(args):
    # Load the trained models
    print("Loading trained models...")
    rf_model = joblib.load('results/random_forest.pkl')
    xgb_model = joblib.load('results/xgboost.pkl')
    
    # Extract features from the input URL
    print(f"Analyzing URL: {args.url}")
    features = extract_features(args.url)
    
    # Create DataFrame with the same structure as training data
    feature_df = pd.DataFrame([features])
    
    # Make predictions
    rf_pred = rf_model.predict_proba(feature_df)[:, 1]
    xgb_pred = xgb_model.predict_proba(feature_df)[:, 1]
    
    # Ensemble prediction (average of both models)
    ensemble_pred = (rf_pred + xgb_pred) / 2
    final_prediction = 1 if ensemble_pred > 0.5 else 0
    
    # Display results
    print("\n" + "="*50)
    print("PHISHING URL DETECTION RESULTS")
    print("="*50)
    print(f"URL: {args.url}")
    print(f"Prediction: {'PHISHING' if final_prediction == 1 else 'LEGITIMATE'}")
    print(f"Confidence: {ensemble_pred[0]:.4f}")
    print(f"Risk Level: {'HIGH' if ensemble_pred[0] > 0.7 else 'MEDIUM' if ensemble_pred[0] > 0.3 else 'LOW'}")
    
    print("\nModel Probabilities:")
    print(f"  Random Forest: {rf_pred[0]:.4f}")
    print(f"  XGBoost: {xgb_pred[0]:.4f}")
    print(f"  Ensemble: {ensemble_pred[0]:.4f}")
    
    print("\nExtracted Features:")
    for feature, value in features.items():
        print(f"  {feature}: {value}")
    
    if final_prediction == 1:
        print("\nWARNING: This URL appears to be PHISHING!")
        print("   Do not enter personal information on this site.")
    else:
        print("\nThis URL appears to be LEGITIMATE.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict if a URL is phishing or legitimate')
    parser.add_argument('--url', required=True, help='URL to analyze')
    args = parser.parse_args()
    main(args)
