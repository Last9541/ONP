import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import zadatak2


def run_logistic_regression():
    df = zadatak2.prepare_games_df()

    df["review_ratio"] = df["Positive"] / (df["Positive"] + df["Negative"])
    features = ["Price","Peak CCU","Required age","DLC count","review_ratio"]

    X = df[features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(df[features].median())
    y_cont = df["Owners_mid"]

    mask = y_cont.notna()
    X = X[mask]
    y_cont = y_cont[mask]

    threshold = y_cont.median()
    y = (y_cont >= threshold).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_scaled,X_test_scaled,scaler=zadatak2.scale_standard(X_train, X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    return {"Accuracy": accuracy_score(y_test, y_pred),"Precision": precision_score(y_test, y_pred),"Recall": recall_score(y_test, y_pred),"F1": f1_score(y_test, y_pred)}


if __name__ == "__main__":
    results = run_logistic_regression()
    print("Logistic Regression results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
