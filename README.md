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

## Model Evaluation

The models are evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-Score

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements.


## Contact

For any questions or suggestions, feel free to contact [your-shrikrishnavyas111@gmail.com](mailto:shrikrishnavyas111@gmail.com).
