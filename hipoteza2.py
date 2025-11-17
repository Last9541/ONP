from scipy import stats
import zadatak2

df = zadatak2.prepare_games_df()
print("HIPOTEZA 2 - Da li se cene igara razlikuju između igara sa popustom i bez popusta?")
prices = df["Price"]
discount = df["Discount"]

group_no_disc = prices[discount == 0].dropna()
group_disc    = prices[discount > 0].dropna()

print("Broj igara bez popusta :", len(group_no_disc))
print("Broj igara sa popustom:", len(group_disc))
print("Srednja cena (bez popusta):", group_no_disc.mean())
print("Srednja cena (sa popustom):", group_disc.mean())

# Shapiro–Wilk za normalnost (uzorak do 5000)
sample_size2 = min(len(group_no_disc), len(group_disc), 5000)

sample_no  = group_no_disc.sample(n=sample_size2, random_state=0)
sample_yes = group_disc.sample(n=sample_size2, random_state=0)

sh_no_stat,  sh_no_p  = stats.shapiro(sample_no)
sh_yes_stat, sh_yes_p = stats.shapiro(sample_yes)

print("\nShapiro–Wilk (bez popusta):  p =", sh_no_p)
print("Shapiro–Wilk (sa popustom):  p =", sh_yes_p)

# ovde je p << 0.05 - distribucije nisu normalne
# t-test se ne koristi, ide Mann–Whitney

# Mann–Whitney U test
u_stat, p_val = stats.mannwhitneyu(group_no_disc, group_disc, alternative="two-sided")

print("\nMann–Whitney U (cena bez vs sa popustom)")
print("U statistika =", u_stat)
print("p-vrednost   =", p_val)

if p_val < 0.05:
    print("Odbacujemo H0: postoji statistički značajna razlika u cenama.")
else:
    print("Ne odbacujemo H0: nema statistički značajne razlike u cenama.")
