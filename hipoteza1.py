from scipy import stats
import zadatak2

df = zadatak2.prepare_games_df()
print("Hipoteza 1 – razlika u cenama Action vs Casual")

action_price = df[df["Primary_genre"] == "Action"]["Price"].dropna()
casual_price = df[df["Primary_genre"] == "Casual"]["Price"].dropna()

print(f"Broj Action igara: {len(action_price)}")
print(f"Broj Casual igara: {len(casual_price)}")
print(f"Srednja cena Action: {action_price.mean():.2f}")
print(f"Srednja cena Casual: {casual_price.mean():.2f}")



# ovde je p << 0.05 - distribucije nisu normalne
# t-test se ne koristi, ide Mann–Whitney

# Mann–Whitney U test
u_stat, p_val = stats.mannwhitneyu(action_price, casual_price, alternative="two-sided")
print(f"Mann–Whitney U = {u_stat:.2f}, p = {p_val:.3e}")

if p_val < 0.05:
    print("Odbacujemo H0: postoji statistički značajna razlika u cenama.")
else:
    print("Ne odbacujemo H0: nema značajne razlike u cenama.")