import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from mlflow.models import infer_signature

mlflow.set_experiment("Breast_Cancer_CI")

print("📌 Mulai training... memastikan file tersedia")

df = pd.read_csv("breast_preprocessed_train.csv")
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

with mlflow.start_run(run_name="CI_BEST_MODEL"):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    # MLflow Signature (hilangkan warning)
    signature = infer_signature(X, model.predict(X))
    example = X.iloc[:5]

    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=example
    )

    mlflow.log_metric("note", 1.0)

    print("🎉 CI Training selesai – Model berhasil di-log ke MLflow!")
