import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

dagshub.auth.add_app_token(os.environ["DAGSHUB_TOKEN"])
dagshub.init(repo_owner='BaniBahariHakim', repo_name='Workflow-CI', mlflow=True)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    file_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "train.csv"
    )

    data = pd.read_csv(file_path)
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("quality_label", axis=1), data["quality_label"],
        test_size=0.2, random_state=42
    )

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth":    [5, 10, 20]
    }

    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=3, scoring="accuracy", n_jobs=-1
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    y_pred = best.predict(X_test)

    existing_run_id = os.environ.get("MLFLOW_RUN_ID")
    with mlflow.start_run(run_id=existing_run_id) as run:
        print(f"Logging to run: {run.info.run_id}")

        precision, recall, fbeta, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
        for i, (p, r, f) in enumerate(zip(precision, recall, fbeta)):
            mlflow.log_metrics({
                f"precision_class_{i}": p,
                f"recall_class_{i}": r,
                f"f1_class_{i}": f
            })

        mlflow.log_metric("macro_precision", precision.mean())
        mlflow.log_metric("macro_recall", recall.mean())
        mlflow.log_metric("macro_f1", fbeta.mean())

        mlflow.log_param("n_estimators",      gs.best_params_["n_estimators"])
        mlflow.log_param("max_depth",         gs.best_params_["max_depth"])
        mlflow.log_param("criterion",         best.criterion)
        mlflow.log_param("max_features",      best.max_features)
        mlflow.log_param("random_state",      42)
        mlflow.log_param("cv_folds",          3)

        mlflow.log_metric("accuracy",          accuracy_score(y_test, y_pred))
        mlflow.log_metric("training_accuracy", best.score(X_train, y_train))
        mlflow.log_metric("best_cv_score",     gs.best_score_)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax)
        plt.tight_layout()
        fig.savefig("confusion_matrix.png")
        plt.close(fig)
        mlflow.log_artifact("confusion_matrix.png")

        fi = dict(zip(X_train.columns.tolist(), best.feature_importances_.tolist()))
        with open("feature_importances.json", "w") as fp:
            json.dump(fi, fp, indent=2)
        mlflow.log_artifact("feature_importances.json")

        mlflow.sklearn.log_model(
            sk_model=best,
            artifact_path="model",
            input_example=X_test.head(5),
            signature=mlflow.models.infer_signature(X_test, y_pred)
        )

        print(f"Model successfully logged to run {run.info.run_id}")