# MLProject/modelling.py
import mlflow
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

mlflow.set_experiment("Breast_Cancer_CI")

df = pd.read_csv("breast_preprocessed_train.csv")
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

with mlflow.start_run(run_name="CI_BEST_MODEL"):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("note", 1.0)  # dummy metric biar run keliatan
    print("CI Training Selesai – Model sudah di-log!")