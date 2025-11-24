# MLProject/modelling.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

def main():
    print("📌 Mulai training... memastikan file tersedia")

    data_path = "breast_preprocessed_train.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File '{data_path}' tidak ditemukan. Pastikan workflow meng-copy file preprocessing.")

    df = pd.read_csv(data_path)
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    mlflow.set_experiment("Breast_Cancer_CI")

    with mlflow.start_run(run_name="CI_BEST_MODEL"):
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
        model.fit(X, y)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Dummy metric agar terlihat di MLflow UI
        mlflow.log_metric("note", 1.0)

        print("🎉 CI Training selesai – Model berhasil di-log ke MLflow!")

if __name__ == "__main__":
    main()
