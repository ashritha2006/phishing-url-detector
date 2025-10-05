import matplotlib
matplotlib.use('Agg')  # ✅ Use non-GUI backend (prevents TclError on Windows)

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap
import joblib
import argparse

def main(args):
    # ✅ Ensure results folder exists
    os.makedirs('results', exist_ok=True)

    # ✅ Load dataset
    df = pd.read_csv(args.data)
    X = df.drop('label', axis=1)
    y = df['label']

    # ✅ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict_proba(X_test)[:, 1]

    # ✅ XGBoost (cleaned warning)
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict_proba(X_test)[:, 1]

    # ✅ Ensemble (average of RF + XGBoost)
    ensemble_pred = (rf_pred + xgb_pred) / 2
    final_labels = (ensemble_pred > 0.5).astype(int)

    # ✅ Metrics calculation
    metrics = {
        'Accuracy': accuracy_score(y_test, final_labels),
        'Precision': precision_score(y_test, final_labels),
        'Recall': recall_score(y_test, final_labels),
        'F1 Score': f1_score(y_test, final_labels),
        'ROC AUC': roc_auc_score(y_test, ensemble_pred)
    }

    # ✅ Save & print metrics
    pd.DataFrame([metrics]).to_csv('results/metrics.csv', index=False)
    print("\nModel Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # ✅ Confusion Matrix Plot
    cm = confusion_matrix(y_test, final_labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()

    # ✅ SHAP Explainability (XGBoost) — optimized for speed and clarity
    if args.shap:
        print("\nGenerating SHAP feature importance plots...")
        import time
        start_time = time.time()
        
        # Use smaller sample size for faster computation
        X_shap = X_test.sample(n=50, random_state=42)
        print(f"   Computing SHAP values for {len(X_shap)} samples...")

        # Use XGBoost explainer (faster than Random Forest)
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_shap)
        
        # 1. Feature Importance Bar Chart (easier to understand)
        print("   Creating feature importance bar chart...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=15)
        plt.title('Feature Importance (SHAP Values)', fontsize=16, fontweight='bold')
        plt.xlabel('Mean |SHAP value| (average impact on model output)', fontsize=12)
        plt.tight_layout()
        plt.savefig('results/shap_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Summary Plot with better formatting
        print("   Creating detailed summary plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_shap, show=False, max_display=15, 
                         plot_size=(12, 8), color_bar_label='Feature Value')
        plt.title('SHAP Summary Plot - Feature Impact on Predictions', fontsize=16, fontweight='bold')
        plt.xlabel('SHAP value (impact on model output)', fontsize=12)
        plt.tight_layout()
        plt.savefig('results/shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Force plot for a single prediction (most interpretable)
        print("   Creating force plot for sample prediction...")
        try:
            # Create a force plot for one sample
            shap.force_plot(explainer.expected_value, shap_values[0], X_shap.iloc[0], 
                           matplotlib=True, show=False)
            plt.title('SHAP Force Plot - How Features Contribute to One Prediction', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('results/shap_force.png', dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"   Note: Force plot skipped due to: {str(e)[:50]}...")
            # Create a simple bar plot of SHAP values for one sample instead
            plt.figure(figsize=(10, 6))
            sample_shap = shap_values[0]
            feature_names = X_shap.columns
            sorted_idx = np.argsort(np.abs(sample_shap))[-15:]  # Top 15 features
            
            plt.barh(range(len(sorted_idx)), sample_shap[sorted_idx])
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.xlabel('SHAP Value')
            plt.title('SHAP Values for Sample Prediction')
            plt.tight_layout()
            plt.savefig('results/shap_sample.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 4. Print top features with their importance
        print("\n   Top 10 Most Important Features:")
        feature_importance = pd.DataFrame({
            'feature': X_shap.columns,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"   {i:2d}. {row['feature']:<25} (importance: {row['importance']:.4f})")
        
        elapsed_time = time.time() - start_time
        print(f"\n   SHAP computation completed in {elapsed_time:.2f} seconds")
        print("   Generated SHAP plots: feature_importance.png, summary.png, and sample/force plot")
    else:
        print("\nSkipping SHAP computation (use --shap to enable)")

    # ✅ Save trained models
    joblib.dump(rf, 'results/random_forest.pkl')
    joblib.dump(xgb, 'results/xgboost.pkl')
    print("\nModels and results saved in the 'results/' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to processed feature CSV file')
    parser.add_argument('--shap', action='store_true', help='Enable SHAP computation (slower but provides feature importance)')
    args = parser.parse_args()
    main(args)
