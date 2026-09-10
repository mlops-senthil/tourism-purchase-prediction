
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from huggingface_hub import HfApi, create_repo, hf_hub_download
import mlflow
import mlflow.sklearn

# Hugging Face Dataset Configuration
DATASET_REPO = "senthil31/tourism-dataset"

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

repo_id = "senthil31/tourism-dataset"
files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]

# Dictionary to store local paths of downloaded files
downloaded_paths = {}

for file in files:
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=file,
        repo_type="dataset",
        token=HF_TOKEN,  # Optional if the repository is public
    )
    downloaded_paths[file] = local_path
    print(f"Downloaded {file} to: {local_path}")

X_train = pd.read_csv(downloaded_paths["X_train.csv"])
X_test = pd.read_csv(downloaded_paths["X_test.csv"])
y_train = pd.read_csv(downloaded_paths["y_train.csv"]).values.ravel()
y_test = pd.read_csv(downloaded_paths["y_test.csv"]).values.ravel()

# Set up MLflow
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Tourism_Product_Prediction_Tuning_Refined")

param_grid = {
    'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'logreg__solver': ['liblinear', 'lbfgs']
}

pipeline_tuned = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(random_state=42, max_iter=1000))
])

grid_search = GridSearchCV(pipeline_tuned, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

print("Starting GridSearchCV...")
grid_search.fit(X_train, y_train)
print("Best parameters: ", grid_search.best_params_)

best_pipeline = grid_search.best_estimator_

with mlflow.start_run(run_name="Logistic_Regression_Tuned_Best_Model"):
    mlflow.log_param("model_type", "Logistic Regression Tuned")
    for param, value in grid_search.best_params_.items():
        mlflow.log_param(param, value)

    y_pred_tuned = best_pipeline.predict(X_test)
    y_pred_proba_tuned = best_pipeline.predict_proba(X_test)[:, 1]

    # Metrics computation
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    precision_tuned = precision_score(y_test, y_pred_tuned, zero_division=0)
    recall_tuned = recall_score(y_test, y_pred_tuned, zero_division=0)
    f1_tuned = f1_score(y_test, y_pred_tuned, zero_division=0)
    roc_auc_tuned = roc_auc_score(y_test, y_pred_proba_tuned)

    mlflow.log_metric("accuracy", accuracy_tuned)
    mlflow.log_metric("precision", precision_tuned)
    mlflow.log_metric("recall", recall_tuned)
    mlflow.log_metric("f1_score", f1_tuned)
    mlflow.log_metric("roc_auc_score", roc_auc_tuned)

    print(f"Tuned Model Accuracy: {accuracy_tuned:.4f}")

    tuned_model_path = "tourism_project/model_building/logistic_regression_tuned_model"
    os.makedirs(tuned_model_path, exist_ok=True)
    joblib.dump(best_pipeline, os.path.join(tuned_model_path, "model.pkl"))
    joblib.dump(X_train.columns.tolist(), os.path.join(tuned_model_path, "model_features.pkl"))

    repo_id_tuned = "senthil31/tourism-product-prediction-tuned-model"

    try:
        create_repo(repo_id=repo_id_tuned, private=False, token=HF_TOKEN, repo_type="model")
    except Exception as e:
        print(f"Repo status: {e}")

    api.upload_folder(folder_path=tuned_model_path, repo_id=repo_id_tuned, repo_type="model", token=HF_TOKEN)
    mlflow.end_run()
