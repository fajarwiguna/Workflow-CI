import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature

# Set experiment (tidak memulai run)
mlflow.set_experiment("Breast_Cancer_CI")

# Load data
df = pd.read_csv("breast_preprocessed_train.csv")
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Log model tanpa start_run()
signature = infer_signature(X, model.predict(X))
mlflow.sklearn.log_model(model, "model", signature=signature)

# Log metrics
mlflow.log_metric("training_accuracy", model.score(X, y))

print("SUCCESS: CI Training Finished")
