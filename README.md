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

The following models were used to predict heart disease:

- **Logistic Regression**: A baseline model for binary classification.
- **Decision Tree**: A non-linear model that creates decision rules based on feature importance.
- **Random Forest**: An ensemble of decision trees to improve accuracy by averaging multiple trees.
- **K-Nearest Neighbors (KNN)**: A proximity-based classifier.
- **Support Vector Machine (SVM)**: A linear classifier that finds the hyperplane separating classes.
- **Bagging Classifier**: An ensemble method that combines multiple versions of a classifier to reduce variance and avoid overfitting.
- **Ada Classifier**: AdaBoost combines weak classifiers to reduce variance and avoid overfitting.
- **Gradient Boosting**: An ensemble boosting algorithm that iteratively improves weak learners.
- **XGBoost**: An efficient and scalable implementation of gradient boosting.

### Model Training

- **Train-Test Split**: The dataset was split into an 80/20 ratio for training and testing.
- **Cross-Validation**: GridSearchCV was used to find the optimal hyperparameters for each model.
- **Metrics Used**: Models were evaluated using Accuracy, Precision, Recall, and F1-score.

### Model Evaluation Metrics

The models were evaluated based on the following metrics:
- **Accuracy**: Percentage of correct predictions.
- **Precision**: The number of true positives divided by the sum of true and false positives.
- **Recall**: The number of true positives divided by the sum of true positives and false negatives.
- **F1-Score**: The harmonic mean of precision and recall.

## Results

The performance of each model is summarized below:

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 82%      | 0.87      | 0.82   | 0.82     |
| Decision Tree         | 68%      | 0.68      | 0.68   | 0.67     |
| Random Forest         | 82%      | 0.85      | 0.85   | 0.82     |
| K-Nearest Neighbors   | 84%      | 0.87      | 0.87   | 0.84     |
| Support Vector Machine| 82%      | 0.85      | 0.82   | 0.82     |
| Bagging Classifier    | 78%      | 0.81      | 0.80   | 0.70     |
| Ada Boosting          | 78%      | 0.78      | 0.79   | 0.78     |
| Gradient Boosting     | 80%      | 0.81      | 0.81   | 0.80     |
| XGBoost               | 78%      | 0.79      | 0.79   | 0.78     |

- **Best Model**: K-Nearest Neighbors achieved the highest accuracy and F1-score, proving to be the best-performing model on this dataset.

## Model Interpretation

Feature importance from the **Random Forest** and **XGBoost** models was analyzed to determine which features (e.g., age, cholesterol levels) had the most influence on heart disease predictions.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements.

## Contact

For any questions or suggestions, feel free to contact [shrikrishnavyas111@gmail.com](mailto:shrikrishnavyas111@gmail.com).
