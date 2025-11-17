

from scipy import stats
import pandas as pd
import zadatak2

print("Test normalnosti (Shapiro–Wilk) na uzorku od 5000 redova:")
num_cols = ["Price", "Peak CCU", "Required age", "DLC count", "Owners_mid"]
df = zadatak2.prepare_games_df()
sample = df[num_cols].sample(n=5000, random_state=0)

for col in num_cols:
    col_data = sample[col].dropna()
    stat, p = stats.shapiro(col_data)
    print(f"{col:12s} -> W = {stat:.4f}, p = {p:.3e}")


# Shapiro–Wilk test pokazuje da nijedna od numeričkih promenljivih ne prati normalnu distribuciju, jer su sve p-vrednosti značajno manje od 0.05.
# Zbog toga se u daljem radu ne koriste parametarski testovi (poput t-testa), već neparametarski testovi — u našem slučaju Mann–Whitney U test.