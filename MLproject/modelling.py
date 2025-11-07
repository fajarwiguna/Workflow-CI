"""
KRITERIA 2 - Basic Model dengan MLflow Autolog
Dataset: Iris Preprocessed
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# Set MLflow tracking URI (local)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris_Basic_Model")

# Load data
df = pd.read_csv(r"C:\Users\ADVAN\Documents\College\Dicoding\MSML\preprocessing\iris_preprocessed_train.csv")
X = df.drop('target', axis=1)
y = df['target']

# Enable autolog
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Basic_RandomForest"):
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict & Evaluate
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    # Log metrics manually (autolog also logs)
    mlflow.log_metric("train_accuracy", accuracy)
    mlflow.log_param("n_estimators", 100)
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print(f"✅ Basic Model Completed!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")