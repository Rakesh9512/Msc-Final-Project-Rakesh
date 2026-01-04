**Data Visualisation and Analysis of Wine Quality Factors**

**Research Question**
How do variations in wine physicochemical properties influence overall wine quality, and which features are most important for predicting wine quality using Machine Learning?

**Project Overview**
This research project analyzes physicochemical properties of Portuguese Red Vinho Verde wines to bridge the gap between laboratory chemistry and sensory perception. By developing accurate machine learning models to predict wine quality from chemical data, this research aims to provide winemakers with tools to adjust fermentation processes in real-time, target specific quality tiers, reduce waste, and maximize profitability.

**Dataset**
Source: Kaggle WineQT Dataset

Download Link: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data

Format: CSV file

Sample Size: 1,143 wine samples

Target Variable: Quality score (3-8 scale)

Dataset Features
The dataset contains 12 physicochemical properties:

Fixed acidity

Volatile acidity

Citric acid

Residual sugar

Chlorides

Free sulfur dioxide

Total sulfur dioxide

Density

pH

Sulphates

Alcohol

Quality (target variable)

Getting Started
Prerequisites
Google account (for Google Colab)

Internet connection

Basic Python knowledge (optional)

Quick Start (Google Colab)
Open Google Colab:

Visit https://colab.research.google.com

Sign in with your Google account

Click "New Notebook"

Upload Dataset:

Click the folder icon in the left sidebar

Click the upload button

Select the downloaded WineQT.csv file

Copy and Run Code:

Copy the entire code from this repository

Paste it into a new cell in Colab

Click the play button or press Shift+Enter to run

Complete Step-by-Step Instructions
Step 1: Set Up Environment
Create a new cell and paste the following to import all required libraries:

python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_auc_score, roc_curve, auc)
from imblearn.over_sampling import SMOTE
from itertools import cycle
import math
Step 2: Load Data
Create another cell and paste:

python
# Load the dataset (update path if needed)
Data_Path = "/content/WineQT.csv"  # Default Colab path
df = pd.read_csv(Data_Path)
print("Data loaded. Shape:", df.shape)
display(df.head())
Important: If you uploaded the file to a different location, update the Data_Path variable.

Step 3: Validate Data
python
# Check dataset structure
print("Columns:", df.columns.tolist())
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDescriptive statistics:")
display(df.describe())
Step 4: Run Complete Analysis
The remaining code sections will:

Perform exploratory data analysis (EDA)

Preprocess the data

Train three machine learning models

Evaluate model performance

Perform hyperparameter tuning

Save results and visualizations

Code Sections Explained
1. Exploratory Data Analysis (EDA)
This section generates visualizations to understand data distributions and relationships:

Histograms for key features (alcohol, volatile acidity, sulphates)

Box plots to identify outliers

Correlation heatmap showing relationships between features

Scatter plots showing feature-quality relationships

2. Data Preprocessing
Converts the 6-class quality problem (scores 3-8) into a 3-class problem:

Class 0 (Low): Quality ≤ 5

Class 1 (Medium): Quality = 6

Class 2 (High): Quality ≥ 7

This simplifies the prediction task while maintaining meaningful quality distinctions.

3. Model Training
Three machine learning models are implemented:

Random Forest Classifier: Ensemble method that provides feature importance scores

XGBoost: Advanced gradient boosting with regularization

Support Vector Machine (SVM): RBF kernel for complex decision boundaries

4. Model Evaluation
Each model is evaluated using:

Accuracy scores

Precision, recall, and F1-scores

Confusion matrices

ROC curves and AUC scores

Comparative performance analysis

5. Hyperparameter Tuning
Random Forest parameters are optimized using:

RandomizedSearchCV for efficient parameter search

Tuned parameters: n_estimators, max_depth, min_samples_split, min_samples_leaf

The best model is saved for potential deployment

6. Results Generation
The code automatically:

Creates a plots/ directory with all visualizations

Saves trained models as .joblib files

Generates classification reports as CSV files

Produces model comparison charts

Expected Results
Key Findings from Analysis
Alcohol shows the strongest positive correlation with quality

Volatile acidity has the strongest negative correlation with quality

Random Forest typically achieves the highest accuracy

Feature importance analysis reveals which chemical properties most influence quality

Generated Outputs
Visualizations (in plots/ directory):

Distribution plots

Correlation heatmaps

Model performance comparisons

Confusion matrices

ROC curves

Saved Models:

scaler.joblib - Data preprocessing scaler

best_random_forest.joblib - Optimized Random Forest model

Performance Reports:

CSV files with detailed classification metrics for each model

Research Significance
Practical Applications
Quality Control: Winemakers can predict quality before sensory testing

Process Optimization: Adjust fermentation based on chemical measurements

Waste Reduction: Identify substandard batches early in production

Profit Maximization: Target specific quality tiers for different market segments

Methodological Contributions
Feature Importance Analysis: Identifies which chemical properties most influence perceived quality

Model Comparison: Evaluates multiple ML algorithms for wine quality prediction

Hyperparameter Optimization: Demonstrates improved performance through systematic tuning

Class Consolidation: Shows effective strategy for handling ordinal quality ratings

References
Cortez, P., Cerdeira, A., Almeida, F., Matos, T. and Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), pp. 547–553.

Kumar, S., Agrawal, K. and Mandan, N. (2022). Addressing class imbalance in wine quality prediction. International Journal of Computer Applications, 183(44), pp. 12–18.

Zhang, C. and Liu, C. (2024). XGBoost for viticulture: a comparative analysis. Computers and Electronics in Agriculture, 205, 107621.

Troubleshooting
Common Issues
File Not Found Error:

Verify WineQT.csv is uploaded to Colab

Check the file path in the Data_Path variable

Import Errors:

Install missing packages: !pip install package_name

Common packages to install: xgboost, imbalanced-learn

Memory Issues in Colab:

Restart runtime: Runtime → Restart runtime

Clear output: Runtime → Restart and run all

Slow Execution:

Use GPU acceleration: Runtime → Change runtime type → GPU

Reduce dataset size for testing (first 100 rows)

Google Colab Tips
Mount Google Drive (if dataset is in Drive):

python
from google.colab import drive
drive.mount('/content/drive')
Data_Path = "/content/drive/MyDrive/WineQT.csv"
Save Outputs to Drive:

python
from google.colab import drive
drive.mount('/content/drive')
# Modify save paths to point to Drive
Download Generated Files:

Right-click files in Colab file browser

Select "Download"

Code Customization
Adjustable Parameters
Class Thresholds: Modify the quality_to_class function to change classification boundaries

SMOTE Usage: Set USE_SMOTE = True to enable class imbalance handling

Test Size: Change test_size in train_test_split (default: 0.20)

Model Parameters: Adjust hyperparameters in each model definition

Extending the Research
Additional Models: Add other classifiers (Neural Networks, KNN, etc.)

Feature Engineering: Create new features from existing measurements

Regression Approach: Predict exact quality scores instead of classes

Ensemble Methods: Combine predictions from multiple models

**Conclusion**
This research project demonstrates that machine learning can effectively predict wine quality from physicochemical properties. The Random Forest classifier, with feature importance analysis, provides both accurate predictions and interpretable insights into which chemical properties most influence wine quality. These findings have practical implications for the wine industry, offering data-driven approaches to quality control and process optimization.

License
This project is for academic and research purposes. Please cite the original dataset and research papers when using this work.
