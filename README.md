# Heart Disease Prediction using Various Machine Learning Algorithms

## Overview

This project aims to predict the presence of heart disease in patients using various machine learning algorithms. The dataset includes multiple health indicators that can be used to classify patients as having or not having heart disease. The analysis involves data exploration, preprocessing, model training, and evaluation to identify the most effective model for prediction.

## Data Description

The dataset used is named `heart.csv` and contains the following features:

- **age**: Age of the patient in years
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type
- **trestbps**: Resting blood pressure (in mm Hg)
- **chol**: Serum cholesterol level (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise-induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: The slope of the peak exercise ST segment
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
- **target**: Diagnosis of heart disease (1 = yes, 0 = no)

## Project Workflow

1. **Data Import and Libraries**:
   - Utilizes libraries such as Pandas, Matplotlib, Seaborn, and Scikit-learn.

2. **Data Loading and Exploration**:
   - The dataset is loaded and inspected for basic statistics and distribution. Exploratory data analysis is performed to understand the relationship between features and the target variable.

3. **Data Visualization**:
   - Visualizations include count plots for the target variable, distributions across different features, and cross-tabulations for categorical variables.

4. **Preprocessing**:
   - Data preprocessing steps include feature scaling and data splitting into training and testing sets.

5. **Model Selection and Training**:
   - Several machine learning models are evaluated, including:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
     - Decision Tree
     - Random Forest
     - Naive Bayes
   - Random Forest emerged as the most effective model based on training and testing accuracy scores.

6. **Model Evaluation**:
   - The selected model (Random Forest) is evaluated using metrics like accuracy, classification report, and confusion matrix.

7. **Predictive Model Deployment**:
   - The model is used to predict heart disease presence for new patient data inputs.

## Key Results

- **Accuracy**: The Random Forest model achieved the highest accuracy on both training and testing datasets.
- **Evaluation Metrics**: The classification report and confusion matrix provided detailed insights into model performance, including precision, recall, and F1 score.

## How to Run

1. Ensure all required libraries are installed.
2. Load the Jupyter notebook in a compatible environment.
3. Execute the cells sequentially to preprocess data, train models, evaluate performance, and make predictions.

## Conclusion

The project successfully identifies the best model for predicting heart disease among the patients. The Random Forest classifier outperformed other algorithms in terms of accuracy and robustness.

## Requirements

The following Python libraries are required:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Acknowledgments

Thanks to the contributors of the dataset and the developers of the open-source tools used in this analysis.

---

This project serves as an educational exercise in machine learning model development and evaluation. The results and models should not be used for clinical decision-making without further validation.
