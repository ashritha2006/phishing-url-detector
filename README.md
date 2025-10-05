# 🛡️ Phishing URL Detector

A machine learning system that detects phishing URLs using ensemble methods (Random Forest + XGBoost) with SHAP explainability.

## 📋 Table of Contents
- [Quick Start](#quick-start)
- [Setup Instructions](#setup-instructions)
- [Usage Commands](#usage-commands)
- [Project Structure](#project-structure)
- [Understanding Results](#understanding-results)
- [Troubleshooting](#troubleshooting)
- [File Descriptions](#file-descriptions)

## 🚀 Quick Start

### 1. **Setup Environment**
```bash
# Navigate to project directory
cd C:\Users\vidiy\Desktop\phishing-url-detector

# Activate virtual environment
venv\Scripts\activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. **Test Everything Works**
```bash
# Test prediction
python src/predict.py --url "https://www.google.com"

# Check models
python src/load_models.py
```

### 3. **Run Phishing Detection**
```bash
# Test with a legitimate URL
python src/predict.py --url "https://www.google.com"

# Test with a suspicious URL
python src/predict.py --url "http://secure-login-verify-account-update.bank-password-signin.com"
```

### 4. **Train New Models (Optional)**
```bash
# Quick training (30 seconds)
python src/train_model.py --data data/processed_features.csv

# Full training with SHAP (2 minutes)
python src/train_model.py --data data/processed_features.csv --shap

# Baseline training
python src/train_model_baseline.py --data data/processed_features.csv
```

## 📁 Project Structure

```
phishing-url-detector/
├── data/                          # Dataset files
│   ├── urls_dataset.csv          # Original dataset
│   ├── processed_features.csv    # Feature-engineered dataset
│   └── improved_features.csv     # Enhanced features
├── results/                      # Model outputs and results
│   ├── *.pkl                     # Trained models (binary files)
│   ├── *.png                     # Visualization plots
│   └── *.csv                     # Performance metrics
├── src/                          # Source code
│   ├── preprocess.py             # Data preprocessing
│   ├── train_model_baseline.py   # Baseline model training
│   ├── train_model.py            # Full model training
│   ├── predict.py                # URL prediction
│   ├── load_models.py             # Model inspection
│   ├── explain_pkl.py            # PKL file explanation
│   └── demo_pkl.py               # PKL demonstration
└── venv/                         # Virtual environment
```

## 🔧 Setup Instructions

### **Prerequisites**
- Python 3.8+
- Windows 10/11
- PowerShell or Command Prompt

### **Environment Setup**
```bash
# 1. Navigate to project directory
cd C:\Users\vidiy\Desktop\phishing-url-detector

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Verify installation
python --version
pip list
```

### **Required Packages**
The virtual environment already contains:
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning
- `xgboost` - Gradient boosting
- `shap` - Model explainability
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `joblib` - Model serialization

## 📊 Usage Commands

### **🎯 Main Usage (URL Prediction)**
```bash
# Basic prediction
python src/predict.py --url "https://example.com"

# Test legitimate URLs
python src/predict.py --url "https://www.google.com"
python src/predict.py --url "https://www.github.com"
python src/predict.py --url "https://www.microsoft.com"

# Test suspicious URLs
python src/predict.py --url "http://fake-bank-login.com"
python src/predict.py --url "http://secure-login-verify-account-update.bank-password-signin.com"
```

### **🤖 Model Training**
```bash
# Quick training (30 seconds)
python src/train_model.py --data data/processed_features.csv

# Full training with SHAP (2 minutes)
python src/train_model.py --data data/processed_features.csv --shap

# Baseline training (XGBoost only)
python src/train_model_baseline.py --data data/processed_features.csv
```

### **🔍 Model Inspection**
```bash
# View model information
python src/load_models.py

# Understand PKL files
python src/explain_pkl.py

# PKL demonstration
python src/demo_pkl.py
```

### **📊 Data Processing**
```bash
# Process raw data into features
python src/preprocess.py
```

### **⚡ Quick Command Reference**
```bash
# Most common commands:
venv\Scripts\activate                                    # Activate environment
python src/predict.py --url "your-url-here"             # Test URL
python src/train_model.py --data data/processed_features.csv --shap  # Train with SHAP
python src/load_models.py                                # Check models
```

## 📈 Understanding Results

### **Prediction Output**
```
==================================================
PHISHING URL DETECTION RESULTS
==================================================
URL: https://example.com
Prediction: LEGITIMATE
Confidence: 0.2054
Risk Level: LOW

Model Probabilities:
  Random Forest: 0.1300
  XGBoost: 0.2808
  Ensemble: 0.2054
```

### **Risk Levels**
- **LOW** (0.0 - 0.3): Likely legitimate
- **MEDIUM** (0.3 - 0.7): Suspicious, investigate further
- **HIGH** (0.7 - 1.0): Likely phishing

### **Performance Metrics**
- **Accuracy**: 85.83%
- **Precision**: 82.59%
- **Recall**: 46.31%
- **F1 Score**: 59.34%
- **ROC AUC**: 84.21%

### **Generated Files**
- `results/metrics.csv` - Performance metrics
- `results/confusion_matrix.png` - Model performance visualization
- `results/shap_*.png` - Feature importance plots (if --shap used)
- `results/*.pkl` - Trained models

## 🔍 Feature Engineering

The model uses 16 features extracted from URLs:

### **Basic Features**
- `url_length` - Total URL length
- `num_dots` - Number of dots in URL
- `subdomain_count` - Number of subdomains
- `has_ip` - Contains IP address
- `has_at` - Contains @ symbol
- `prefix_suffix` - Has http/https prefix
- `https_in_domain` - HTTPS in domain name
- `entropy` - Character randomness

### **Keyword Features**
- `kw_login` - Contains "login"
- `kw_secure` - Contains "secure"
- `kw_update` - Contains "update"
- `kw_verify` - Contains "verify"
- `kw_bank` - Contains "bank"
- `kw_account` - Contains "account"
- `kw_signin` - Contains "signin"
- `kw_password` - Contains "password"

## 🛠️ Troubleshooting

### **Common Issues**

#### **1. Virtual Environment Not Activated**
```bash
# Error: ModuleNotFoundError
# Solution: Activate virtual environment
venv\Scripts\activate
```

#### **2. Unicode Errors**
```bash
# Error: UnicodeEncodeError
# Solution: Scripts are fixed to avoid emoji characters
```

#### **3. SHAP Taking Too Long**
```bash
# Solution: Use without --shap flag for faster training
python src/train_model.py --data data/processed_features.csv
```

#### **4. PKL Files Not Opening**
```bash
# PKL files are binary - don't open as text
# Use: python src/predict.py --url "your-url"
```

### **Performance Optimization**
- **Without SHAP**: ~30 seconds training
- **With SHAP**: ~2 minutes training
- **Prediction**: <1 second per URL

## 📚 File Descriptions

### **Core Scripts**
- `predict.py` - Main prediction script
- `train_model.py` - Full model training with ensemble
- `train_model_baseline.py` - Baseline model training
- `preprocess.py` - Data preprocessing

### **Utility Scripts**
- `load_models.py` - Inspect trained models
- `explain_pkl.py` - Understand PKL files
- `demo_pkl.py` - PKL file demonstration

### **Data Files**
- `urls_dataset.csv` - Original dataset
- `processed_features.csv` - Feature-engineered data
- `improved_features.csv` - Enhanced features

### **Result Files**
- `*.pkl` - Trained models (binary)
- `*.png` - Visualization plots
- `*.csv` - Performance metrics

## 🎯 Key Features

- **Ensemble Learning**: Random Forest + XGBoost
- **High Accuracy**: 85.83% accuracy
- **Fast Prediction**: <1 second per URL
- **Feature Importance**: SHAP explainability
- **Easy to Use**: Simple command-line interface
- **Comprehensive**: 16 engineered features

## 📞 Support

If you encounter issues:
1. Check virtual environment is activated
2. Verify all dependencies are installed
3. Check file paths are correct
4. Review error messages for specific issues

## 🚀 Next Steps

1. **Test with your own URLs**
2. **Experiment with different models**
3. **Add new features**
4. **Deploy to production**
5. **Create web interface**
## 🔗 Pretrained Models

The trained model files (`.pkl`) are **too large to store directly on GitHub** (GitHub has a 100 MB file size limit).  
To make the project reproducible, the models have been uploaded separately:

- **Random Forest Model**: [Download Link](https://drive.google.com/file/d/1u4ckPcXjw3-pCbjB4E_SsPkqpsYNoZVq/view?usp=sharing)  
- **XGBoost Model**: [Download Link](https://drive.google.com/file/d/1R2lnsg-aFkkcUxsEMFDih7OW01s-U1jA/view?usp=sharing)  

📂 Place the downloaded `.pkl` files into the `results/` folder before running `predict.py` or `load_models.py`.  



