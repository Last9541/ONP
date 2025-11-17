import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA



def scale_standard(df, columns):
    scaler = StandardScaler()
    return scaler.fit_transform(df[columns]), scaler

def normalize_minmax(df, columns):
    scaler = MinMaxScaler()
    return scaler.fit_transform(df[columns]), scaler

def normalize_l2(df, columns):
    norm = Normalizer(norm='l2')
    return norm.fit_transform(df[columns])

def normalize_l1(df, columns):
    norm = Normalizer(norm='l1')
    return norm.fit_transform(df[columns])


def parse_owners(s):
    if isinstance(s, str) and " - " in s:
        low, high = s.split(" - ")
        try:
            low = int(low.replace(",", ""))
            high = int(high.replace(",", ""))
            return (low + high) / 2
        except ValueError:
            return np.nan
    return np.nan


def primary_genre(s):
    if isinstance(s, str) and s.strip():
        return s.split(",")[0].strip()
    return np.nan


def prepare_games_df(do_print=False,csv_path: str = "games.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Owners_mid"] = df["Estimated owners"].apply(parse_owners)
    df["Primary_genre"] = df["Genres"].apply(primary_genre)


    #Na osnovu ovoga smo zakljucili da kolona Primary_genre ima nedostajuce vrednosti
    if do_print:
        print("\nBroj nedostajućih vrednosti u ključnim kolonama:")
        num_cols1 = ["Price", "Peak CCU", "Required age", "DLC count", "Owners_mid"]
        cat_cols1 = ["Primary_genre"]
        missing = df[num_cols1 + cat_cols1].isna().sum()
        print(missing)


    df["Primary_genre"] = df["Primary_genre"].fillna("Unknown")
    return df

def price_plot(df):
    col = "Price"
    data = df[col].dropna()

    upper = data.quantile(0.99)
    data_trunc = data[data <= upper]

    plt.figure(figsize=(8, 5))
    plt.hist(data_trunc, bins=50, edgecolor="black")
    plt.title("Histogram – Price (bez gornjih 1% outliera)")
    plt.xlabel("Price")
    plt.ylabel("Frekvencija")
    plt.tight_layout()
    plt.show()


def dlc_count_plot(df):
    col = "DLC count"
    data = df[col].dropna()

    upper = data.quantile(0.99)
    data_trunc = data[data <= upper]

    plt.figure(figsize=(8, 5))
    plt.hist(data_trunc, bins=50, edgecolor="black")
    plt.title("Histogram – DLC count (bez gornjih 1% outliera)")
    plt.xlabel("DLC count")
    plt.ylabel("Frekvencija")
    plt.tight_layout()
    plt.show()

def owners_mid_plot(df):
    col = "Owners_mid"
    data = df[col].dropna()
    data = data[data > 0]  # izbacujemo igre sa 0 vlasnika

    bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 50)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, edgecolor="black")
    plt.xscale("log")
    plt.title("Histogram – Owners_mid (log-log)")
    plt.xlabel("Owners_mid")
    plt.ylabel("Frekvencija")
    plt.tight_layout()
    plt.show()


def scale(df,num_cols):
    X = df[num_cols].copy()
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def pca_method(x_scaled):
    pca = PCA()
    X_pca = pca.fit_transform(x_scaled)

    explained_var = pca.explained_variance_ratio_
    cum_explained_var = explained_var.cumsum()
    print("\nPCA – udeo objašnjene varijanse po komponentama:")
    for i, (v, cv) in enumerate(zip(explained_var, cum_explained_var), start=1):
        print(f"Komponenta {i}: {v:.4f} (kumulativno: {cv:.4f})")
    return (X_pca,explained_var,cum_explained_var)

def peak_ccu_plot(df):
    col = "Peak CCU"
    data = df[col].dropna()
    data = data[data > 0]

    bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 50)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, edgecolor="black")
    plt.xscale("log")
    plt.title("Histogram – Peak CCU (log-log)")
    plt.xlabel("Peak CCU")
    plt.ylabel("Frekvencija")
    plt.tight_layout()
    plt.show()


def required_age_plot(df):
    plt.figure(figsize=(8, 5))
    s = df["Required age"]
    s.value_counts().sort_index().plot(kind="bar")
    plt.title("Distribucija – Required age (uključujući 0)")
    plt.xlabel("Age")
    plt.ylabel("Broj igara")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    s = s[s > 0]
    s.value_counts().sort_index().plot(kind="bar")
    plt.title("Distribucija – Required age bez 0")
    plt.xlabel("Age")
    plt.ylabel("Broj igara")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = prepare_games_df(True)
    print("\nInfo:")
    print(df.info())
    num_cols = ["Price", "Peak CCU", "Required age", "DLC count", "Owners_mid"]
    cat_cols = ["Primary_genre"]

    print("\nDeskriptivna statistika numeričkih promenljivih:")
    desc = df[num_cols].describe()
    print(desc)

    price_plot(df)
    dlc_count_plot(df)
    owners_mid_plot(df)
    peak_ccu_plot(df)
    required_age_plot(df)

    corr_matrix = df[num_cols].corr()
    print("\nKorelaciona matrica:")
    print(corr_matrix)

    plt.figure(figsize=(6, 5))
    plt.imshow(corr_matrix, interpolation="nearest")
    plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
    plt.yticks(range(len(num_cols)), num_cols)
    plt.colorbar()
    plt.title("Korelaciona matrica (numeričke promenljive)")
    plt.tight_layout()
    plt.show()

    X_pca,explained_var,cum_explained_var=pca_method(scale(df,num_cols))
    plt.figure()
    plt.plot(range(1, len(explained_var) + 1), cum_explained_var, marker="o")
    plt.xlabel("Broj komponenti")
    plt.ylabel("Kumulativni udeo objašnjene varijanse")
    plt.title("PCA – kumulativna objašnjena varijansa")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5, alpha=0.3)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA – prve dve glavne komponente")
    plt.tight_layout()
    plt.show()