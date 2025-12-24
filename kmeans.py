import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import zadatak2




def run_kmeans(k=3):
    df = zadatak2.prepare_games_df()

    df["review_ratio"] = df["Positive"] / (df["Positive"] + df["Negative"])

    features = ["Price", "Peak CCU", "Required age", "DLC count", "review_ratio"]
    X = df[features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(df[features].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)


    df_out = df.copy()
    df_out["cluster"] = labels

    return {"model": model, "scaler": scaler, "labels": labels, "silhouette": float(sil), "features": features, "df_with_clusters": df_out,}

if __name__ == "__main__":
    r = run_kmeans()
    print("KMEANS results:")
    features = r["features"]
    dfc = r["df_with_clusters"]
    print(dfc[["Name", "Price", "Peak CCU", "review_ratio", "cluster"]].head(10))
    print(dfc.groupby("cluster")[features].mean())
