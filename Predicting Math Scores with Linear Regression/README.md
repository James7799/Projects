Math Score Prediction using Machine Learning

Project Overview

This project aims to predict students' math scores based on various features such as reading and writing scores, parental education, gender, ethnicity, and other attributes. The dataset used is the "exams" dataset, which contains students' exam performance details.

Objectives

Download and preprocess the dataset.

Implement different machine learning models to predict math scores.

Compare model performances and suggest improvements.

Technologies Used

Python

Pandas

Scikit-learn

NumPy

Matplotlib

Dataset Information

Source: Downloaded from Google Drive using a script.

Features:

gender

race/ethnicity

parental level of education

lunch

test preparation course

reading score

writing score

Target Variable:

math score

Project Workflow

1. Downloading the Dataset

The function data_yuklab_olish() is used to download the dataset from Google Drive and save it in a specified directory.

2. Data Preprocessing

Feature Selection: Numerical (reading score, writing score) and categorical (gender, race/ethnicity, etc.) features are separated.

Data Splitting: The dataset is divided into training (90%) and test (10%) sets.

Feature Scaling: Standardization is applied to numerical features.

One-Hot Encoding: Categorical features are transformed into numerical values.

3. Model Training

Linear Regression Model: Implemented using Pipeline for preprocessing and training.

Evaluation: Model achieved an R² score of 0.856, indicating strong performance.

4. Potential Model Improvements

Implement Decision Trees or Random Forests to capture nonlinear relationships.

Experiment with a Neural Network (MLP) for better predictions.

Perform feature engineering to extract more useful information.

Results & Findings

Linear Regression performed well but may not be the best model.

The dataset contains categorical variables that may impact performance if not properly encoded.

Feature importance analysis can help refine the model.

Future Work

Implement and compare Decision Tree, Random Forest, and Neural Network models.

Apply hyperparameter tuning for better performance.

Visualize feature importance and correlations.

How to Run the Project

Install dependencies:

pip install pandas numpy scikit-learn matplotlib

Run the script to download and preprocess data.

Train and evaluate the model.

Conclusion

This project provides a foundation for predicting students' math scores using machine learning. Future enhancements will improve model accuracy and interpretability.
