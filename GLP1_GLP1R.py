# GLP1_GLP1R.py

import numpy as np
import matplotlib.pyplot as plt

def Fact(GLP1, S, Kd_GLP1, Kd_S):
    """Fraction of GLP1R activated by either GLP1 or semaglutide."""
    # Equilibrium binding: the two agonists compete for one binding site.
    # Either bound agonist activates the receptor; free ligand stays fixed.
    # Total receptor = free receptor + GLP1-bound + semaglutide-bound.
    return (GLP1 / Kd_GLP1 + S / Kd_S) / (1 + GLP1 / Kd_GLP1 + S / Kd_S)

Kd_GLP1 = 0.17                 # GLP1 dissociation constant [nM]
Kd_S = 0.38                    # semaglutide dissociation constant [nM]
GLP1_fasting = 0.015           # fasting GLP1 [nM]
GLP1_fed = 0.040               # GLP1 after a meal [nM]
S = np.logspace(-4, 4, 500)    # semaglutide concentration [nM]

plt.figure()
plt.semilogx(S, Fact(GLP1_fasting, S, Kd_GLP1, Kd_S), lw=2,
             label="Fasting: [GLP1] = 0.015 nM")
plt.semilogx(S, Fact(GLP1_fed, S, Kd_GLP1, Kd_S), lw=2,
             label="Fed: [GLP1] = 0.040 nM")
plt.axvline(1, ls="--", c="gray", lw=1, label="Semaglutide = 1 nM")
plt.ylim(0, 1)
plt.xlabel("[Semaglutide] (nM, log scale)")
plt.ylabel("Fraction active (Fact)")
plt.title("GLP1 and semaglutide co-activation of GLP1R")
plt.legend()
plt.tight_layout()

# Calculate the baseline separately because zero is not on a log axis.
print("GLP1 alone (no semaglutide):")
print("Fasting:", round(100 * Fact(GLP1_fasting, 0, Kd_GLP1, Kd_S), 2), "%")
print("Fed:", round(100 * Fact(GLP1_fed, 0, Kd_GLP1, Kd_S), 2), "%")
print("With 1 nM semaglutide:")
print("Fasting:", round(100 * Fact(GLP1_fasting, 1, Kd_GLP1, Kd_S), 2), "%")
print("Fed:", round(100 * Fact(GLP1_fed, 1, Kd_GLP1, Kd_S), 2), "%")

# Interpretation: Increasing semaglutide raises GLP1R activation toward 100%
# in both conditions, with fed GLP1 producing greater activation than fasting
# GLP1 (19.05% vs. 8.11% without semaglutide, and 74.14% vs. 73.12% at 1 nM
# semaglutide), so the difference between the curves shrinks as the receptor
# population saturates.

plt.show()
