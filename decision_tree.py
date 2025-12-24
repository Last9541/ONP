import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import zadatak2





def run_decision_tree():
    df = zadatak2.prepare_games_df()

    df["review_ratio"] = df["Positive"] / (df["Positive"] + df["Negative"])

    features = ["Price", "Peak CCU", "Required age", "DLC count", "review_ratio"]
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

    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model, "accuracy": float(accuracy_score(y_test, y_pred)), "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)), "f1": float(f1_score(y_test, y_pred)), "features": list(X.columns),
    }

if __name__ == "__main__":
    r = run_decision_tree()
    print("Decision tree results:")
    print(f"accuracy : {r['accuracy']:.4f}")
    print(f"precision: {r['precision']:.4f}")
    print(f"recall   : {r['recall']:.4f}")
    print(f"f1       : {r['f1']:.4f}")

