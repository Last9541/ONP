from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

import zadatak2
def run_linear_regression():
    df = zadatak2.prepare_games_df()

    df["review_ratio"] = df["Positive"] / (df["Positive"] + df["Negative"])
    features = ["Price","Peak CCU","Required age","DLC count","review_ratio"]

    X = df[features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(df[features].median())
    y = df["Owners_mid"]

    mask = y.notna()
    X = X[mask]
    y = y[mask]

    y = np.log(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_scaled,X_test_scaled,scaler=zadatak2.scale_standard(X_train, X_test)


    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {"MSE": mse, "RMSE": rmse,"R2": r2}


if __name__ == "__main__":
    results = run_linear_regression()
    print("Linear Regression results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")