import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers

import zadatak2





def run_mlp():
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
    X_train_scaled,X_test_scaled,scaler = zadatak2.scale_standard(X_train, X_test)

    # model = MLPClassifier(
    #     hidden_layer_sizes=(64, 32),
    #     activation="relu",
    #     solver="adam",
    #     max_iter=600,
    #     random_state=42,
    #     early_stopping=True,
    #     n_iter_no_change=15
    # )
    model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )
    model.fit(
        X_train_scaled,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[callback],
        verbose=1
    )

    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob >= 0.5).astype(int).ravel()

    # model.fit(X_train_scaled, y_train)
    # y_pred = model.predict(X_test_scaled)

    return {"model": model, "scaler": scaler, "accuracy": float(accuracy_score(y_test, y_pred)), "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)), "f1": float(f1_score(y_test, y_pred)), "features": list(X.columns),}

if __name__ == "__main__":
    r = run_mlp()
    print("MLP results:")
    print(f"accuracy : {r['accuracy']:.4f}")
    print(f"precision: {r['precision']:.4f}")
    print(f"recall   : {r['recall']:.4f}")
    print(f"f1       : {r['f1']:.4f}")

