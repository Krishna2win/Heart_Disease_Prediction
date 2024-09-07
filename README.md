# Heart Disease Prediction

## Overview

This project aims to predict whether a person has heart disease based on various medical attributes using machine learning algorithms. The model is trained on a heart disease dataset and uses features like age, cholesterol levels, resting blood pressure, and more to predict the likelihood of heart disease.

## Features

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Model Training and Tuning (Logistic Regression, Decision Tree, Random Forest, etc.)
- Model Evaluation using Accuracy, Precision, Recall, and F1-score
- Heart Disease Prediction based on new inputs

## Dataset

The dataset used in this project includes the following features:
- **Age**: The age of the individual
- **Sex**: Gender (1 = male, 0 = female)
- **Chest Pain Type**: 0-3 values representing different types of chest pain
- **Resting Blood Pressure**: Blood pressure in mm Hg
- **Cholesterol**: Serum cholesterol in mg/dl
- **Fasting Blood Sugar**: 1 if fasting blood sugar > 120 mg/dl, 0 otherwise
- **Resting ECG Results**: ECG results with values of 0-2
- **Max Heart Rate Achieved**
- **Exercise Induced Angina**: 1 = yes, 0 = no
- **Oldpeak**: ST depression induced by exercise relative to rest
- **Slope**: The slope of the peak exercise ST segment
- **Thal**: 3 = normal, 6 = fixed defect, 7 = reversible defect
- **Target**: 1 = heart disease, 0 = no heart disease

## Getting Started

### Prerequisites

- Python 3.x
- Libraries: Pandas, Numpy, Scikit-learn, Matplotlib, Seaborn

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Krishna2win/Heart_Disease_Prediction.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Heart_Disease_Prediction
    ```

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Place your dataset in the `data/` directory.

2. Run the analysis script:
    ```bash
    python heart_disease_analysis.py
    ```

3. View the results and visualizations in the `results/` folder.

## Files

- `heart_disease_analysis.py`: Script for data preprocessing, model training, and evaluation.
- `requirements.txt`: List of Python dependencies required for the project.
- `data/`: Directory to store dataset files.
- `results/`: Directory where the output (model performance, visualizations) will be saved.

## Modeling

Several machine learning models are used in this project to predict heart disease. The models include:

- **Logistic Regression**: A linear model that predicts probabilities for binary classification.
- **Decision Tree**: A non-linear model that splits the data based on feature importance to make predictions.
- **Random Forest**: An ensemble method that builds multiple decision trees and averages their results.
- **K-Nearest Neighbors (KNN)**: A simple algorithm that classifies a data point based on the majority label of its k-nearest neighbors.

### Model Training

The data was split into training and testing sets using an 80/20 ratio. Hyperparameter tuning was performed using GridSearchCV or RandomizedSearchCV, depending on the model, to find the optimal parameters.

### Model Evaluation Metrics

The models were evaluated based on the following metrics:
- **Accuracy**: Percentage of correct predictions.
- **Precision**: The number of true positives divided by the sum of true and false positives.
- **Recall**: The number of true positives divided by the sum of true positives and false negatives.
- **F1-Score**: The harmonic mean of precision and recall.

## Results

The following are the results from the models:

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 85%      | 0.84      | 0.86   | 0.85     |
| Decision Tree         | 79%      | 0.77      | 0.81   | 0.79     |
| Random Forest         | 88%      | 0.87      | 0.89   | 0.88     |
| K-Nearest Neighbors   | 82%      | 0.80      | 0.83   | 0.81     |

- **Random Forest** provided the highest accuracy and F1-score, making it the best-performing model for this dataset.

## Model Interpretation

Feature importance was analyzed for models like Random Forest, highlighting which features (age, cholesterol levels, etc.) had the most influence on the predictions.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements.

## Contact

For any questions or suggestions, feel free to contact [shrikrishnavyas111@gmail.com](mailto:shrikrishnavyas111@gmail.com).
