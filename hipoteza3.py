from scipy import stats
import zadatak2

df = zadatak2.prepare_games_df()
print("HIPOTEZA 3 – Razlika u cenama između 18+ i non-adult igara")
age = df["Required age"]
prices = df["Price"]

group_nonadult = prices[age == 0].dropna()
group_adult    = prices[age >= 18].dropna()

print("Broj igara bez ograničenja (age=0):", len(group_nonadult))
print("Broj igara 18+ (age>=18):          ", len(group_adult))
print("Srednja cena (age=0): ", group_nonadult.mean())
print("Srednja cena (age>=18):", group_adult.mean())

# Shapiro–Wilk za normalnost (uzorak do 5000)
sample_size3 = min(len(group_nonadult), len(group_adult), 5000)

sample_non = group_nonadult.sample(n=sample_size3, random_state=1)
sample_ad  = group_adult.sample(n=sample_size3, random_state=1)

sh_non_stat, sh_non_p = stats.shapiro(sample_non)
sh_ad_stat,  sh_ad_p  = stats.shapiro(sample_ad)

print("\nShapiro–Wilk (age=0):   p =", sh_non_p)
print("Shapiro–Wilk (age>=18): p =", sh_ad_p)

# ovde je p << 0.05 - distribucije nisu normalne
# t-test se ne koristi, ide Mann–Whitney

# Mann–Whitney U test
u_stat, p_val = stats.mannwhitneyu(group_nonadult, group_adult, alternative="two-sided")

print("\nMann–Whitney U (cena: age=0 vs age>=18)")
print("U statistika =", u_stat)
print("p-vrednost   =", p_val)

if p_val < 0.05:
    print("Odbacujemo H0: postoji statistički značajna razlika u cenama.")
else:
    print("Ne odbacujemo H0: nema statistički značajne razlike u cenama.")
