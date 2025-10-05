import matplotlib
matplotlib.use('Agg')  # ✅ Prevent TclError on Windows

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
import argparse

def main(args):
    # ✅ Ensure results folder exists
    os.makedirs('results', exist_ok=True)

    # ✅ Load baseline feature dataset
    df = pd.read_csv(args.data)
    X = df.drop('label', axis=1)
    y = df['label']

    # ✅ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ Baseline Model — XGBoost only
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)

    # ✅ Metrics calculation
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred_prob)
    }

    # ✅ Save & print metrics
    pd.DataFrame([metrics]).to_csv('results/baseline_metrics.csv', index=False)
    print("\n📊 Baseline Model Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # ✅ Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Baseline Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('results/baseline_confusion_matrix.png')
    plt.close()

    print("\n✅ Baseline results saved in 'results/' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to baseline feature CSV')
    args = parser.parse_args()
    main(args)
