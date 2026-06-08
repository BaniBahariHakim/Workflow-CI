import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import sys
import os

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth    = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    file_path    = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "train.csv"
    )

    data = pd.read_csv(file_path)
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("quality_label", axis=1), data["quality_label"],
        test_size=0.2, random_state=42
    )

    mlflow.sklearn.autolog()

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)
