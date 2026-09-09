
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV # Required for tuning
import joblib
import os
from huggingface_hub import HfApi, create_repo

import mlflow
import mlflow.sklearn

# Assuming X_train, X_test, y_train, y_test are available from previous data preparation step
# If this script is run standalone, you would need to load them here.
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('X_test.csv')

# Set up MLflow for tracking this specific tuning run
# Using sqlite for local tracking within the script context
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Tourism_Product_Prediction_Tuning_Refined")

# Define the parameter grid for Logistic Regression tuning
# 'C' is the inverse of regularization strength; smaller values specify stronger regularization.
# 'solver' specifies the algorithm to use in the optimization problem.
param_grid = {
    'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'logreg__solver': ['liblinear', 'lbfgs'] # 'liblinear' works well for small datasets and L1/L2 regularization
}

# Create the pipeline with StandardScaler for feature scaling and LogisticRegression
pipeline_tuned = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(random_state=42, max_iter=1000)) # Increased max_iter for convergence
])

print("Model and parameter grid defined.")

# Initialize GridSearchCV with the pipeline and parameter grid
# cv=5 means 5-fold cross-validation
# scoring='accuracy' is used to evaluate the models
# n_jobs=-1 uses all available CPU cores
grid_search = GridSearchCV(pipeline_tuned, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Fit GridSearchCV to the training data to perform the tuning
print("Starting GridSearchCV for Logistic Regression tuning...")
grid_search.fit(X_train, y_train)
print("GridSearchCV completed. Best parameters found: ", grid_search.best_params_)

# Get the best estimator (model) found by GridSearchCV
best_pipeline = grid_search.best_estimator_

# Start an MLflow run to log the details of the tuned model and its performance
with mlflow.start_run(run_name="Logistic_Regression_Tuned_Best_Model"):
    # Log general information about the model
    mlflow.log_param("model_type", "Logistic Regression Tuned")

    # Log the best parameters found by GridSearchCV
    for param, value in grid_search.best_params_.items():
        mlflow.log_param(param, value)

    print("Tuned parameters logged to MLflow.")

    # Make predictions with the best tuned model on the test set
    y_pred_tuned = best_pipeline.predict(X_test)
    y_pred_proba_tuned = best_pipeline.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    precision_tuned = precision_score(y_test, y_pred_tuned, zero_division=0)
    recall_tuned = recall_score(y_test, y_pred_tuned, zero_division=0)
    f1_tuned = f1_score(y_test, y_pred_tuned, zero_division=0)
    roc_auc_tuned = roc_auc_score(y_test, y_pred_proba_tuned)

    # Log metrics to the active MLflow run
    mlflow.log_metric("accuracy", accuracy_tuned)
    mlflow.log_metric("precision", precision_tuned)
    mlflow.log_metric("recall", recall_tuned)
    mlflow.log_metric("f1_score", f1_tuned)
    mlflow.log_metric("roc_auc_score", roc_auc_tuned)

    print("Model performance evaluated and metrics logged to MLflow.")
    print(f"Tuned Model Accuracy: {accuracy_tuned:.4f}")
    print(f"Tuned Model Precision: {precision_tuned:.4f}")
    print(f"Tuned Model Recall: {recall_tuned:.4f}")
    print(f"Tuned Model F1-Score: {f1_tuned:.4f}")
    print(f"Tuned Model ROC AUC Score: {roc_auc_tuned:.4f}")

    # Save the best pipeline and feature list locally
    tuned_model_path = "tourism_project/model_building/logistic_regression_tuned_model"
    os.makedirs(tuned_model_path, exist_ok=True)
    joblib.dump(best_pipeline, os.path.join(tuned_model_path, "model.pkl"))
    joblib.dump(X_train.columns.tolist(), os.path.join(tuned_model_path, "model_features.pkl"))
    print(f"Best tuned model saved locally at {tuned_model_path}")


    # api = HfApi(token=os.getenv("HF_TOKEN"))
    HF_TOKEN = os.getenv("HF_TOKEN")
    api = HfApi(token=HF_TOKEN)
    repo_id_tuned = "senthil31/tourism-product-prediction-tuned-model" # A new repo for the tuned model
    api = HfApi()

    try:
        # Attempt to create the repository; if it exists, an exception will be caught
        create_repo(repo_id=repo_id_tuned, private=False, token=HF_TOKEN, repo_type="model")
        print(f"Created new Hugging Face model repo for tuned model: {repo_id_tuned}")
    except Exception as e:
        print(f"Could not create repo for tuned model (might already exist): {e}")

    # Upload the saved model files to the Hugging Face Hub
    api.upload_folder(
        folder_path=tuned_model_path,
        repo_id=repo_id_tuned,
        repo_type="model",
        token=HF_TOKEN
    )
    print(f"Tuned model uploaded to Hugging Face Hub: https://huggingface.co/{repo_id_tuned}")

    # End the MLflow run explicitly
    mlflow.end_run()
