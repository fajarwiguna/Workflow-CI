import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature

mlflow.set_experiment("Breast_Cancer_CI")

df = pd.read_csv("breast_preprocessed_train.csv")
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

with mlflow.start_run(run_name="CI_AUTO_SUCCESS"):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    # Log model + signature
    signature = infer_signature(X, model.predict(X))
    mlflow.sklearn.log_model(model, "model", signature=signature)
    
    mlflow.log_metric("status", 1.0)
    print("CI TRAINING SELESAI – MODEL SUDAH DI-LOG!")