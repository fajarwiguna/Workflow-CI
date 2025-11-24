import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature

# Load data
df = pd.read_csv("breast_preprocessed_train.csv")
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Log model (tidak pakai start_run, tidak pakai set_experiment)
signature = infer_signature(X, model.predict(X))
mlflow.sklearn.log_model(model, "model", signature=signature)

# Log metrics
mlflow.log_metric("training_accuracy", model.score(X, y))

print("SUCCESS: CI Training Finished")
