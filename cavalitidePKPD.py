# cavalitidePKPD.py
# Combined PKPD model: the Cavalitide PK model (cavalitidePK.py) with the ligand-receptor
# model from Task 1 embedded, to predict the fraction of active GLP1R during treatment.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----- PK parameters: Cavalitide daily pill -----
Favailable = 0.01   # bioavailability
ka = 0.006          # 1/h, absorption rate
D = 28.5            # mg per dose; 21 mg gives ~0.5 nM free in blood, 28.5 mg gives ~90% active GLP1R
interval = 24       # h between doses (daily)
n_doses = 100       # number of doses
depot = 'gut (oral)'

ke = 0.0039         # 1/h, elimination rate
V = 10.8            # L
MW = 4113.6         # g/mol
Ffree = 0.01        # rest of drug bound to albumin
dt = 0.1            # h, spacing of the output time points

# ----- PD parameters [nM] -----
Kd1 = 0.17          # L1, GLP1
Kd2 = 0.077         # L2, Cavalitide (optimized Kd from Task 1)
L1 = 0.015          # GLP1 (fasting)

def Fact(L1, L2, Kd1, Kd2):
    """Fractional receptor activation with two agonists L1 and L2"""
    return (Kd2*L1+Kd1*L2) / (Kd1*Kd2 + Kd2*L1 + Kd1*L2)

def ODEfunc(t, y):
    Adepot, Cc = y
    dAdepot = -ka*Adepot            # Adepot: depot amount (mg)
    dCc = ka*Adepot/V - ke*Cc       # Cc: blood concentration (mg/L)
    return [dAdepot, dCc]

def simulate(D, n_doses, interval, dt):
    """Repeated dosing every `interval` h; returns time (h), depot amount (mg),
    blood concentration (nmol/L) and fraction of active GLP1R."""
    t_out, Adepot_out, Cc_out = [], [], []
    y = np.array([0.0, 0.0])
    n_steps = int(round(interval / dt))
    for i in range(n_doses):
        y[0] += Favailable*D                 # absorbed fraction of dose enters the depot
        t0 = i * interval
        t_seg = t0 + dt * np.arange(n_steps + 1)
        sol = solve_ivp(ODEfunc, (t_seg[0], t_seg[-1]), y, t_eval=t_seg, rtol=1e-8, atol=1e-10)
        t_out.append(sol.t)
        Adepot_out.append(sol.y[0])
        Cc_out.append(sol.y[1])
        y = sol.y[:, -1].copy()              # carry state into next interval
    t = np.concatenate(t_out)
    Adepot_out = np.concatenate(Adepot_out)
    Cc_nM = np.concatenate(Cc_out) * 1e-3 / MW * 1e9
    L2 = Ffree*Cc_nM                         # free Cavalitide in blood is the ligand
    return t, Adepot_out, Cc_nM, Fact(L1, L2, Kd1, Kd2)

if __name__ == "__main__":
    # --- run ---
    t, Adepot, Cc, Fa = simulate(D, n_doses, interval, dt)

    last = t >= t[-1] - interval
    print(f"Cavalitide {D} mg every {interval} h: free in blood = {Ffree*Cc[last].mean():.3f} nM, "
          f"fraction of active GLP1R at steady state = {Fa[last].mean():.3f}")

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t / 24, Adepot, label=f"{depot} depot [mg]")
    axes[0].plot(t / 24, Ffree*Cc, label="free in blood [nM]")
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel("Cavalitide")
    axes[0].set_title(f"Cavalitide: {D} mg every {interval} h, {n_doses} doses")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t / 24, Fa, label="Cavalitide + GLP1 (fasting)")
    axes[1].axhline(Fact(L1, 0, Kd1, Kd2), ls="--", c="gray", label="GLP1 alone (fasting)")
    axes[1].axhline(0.9, ls=":", c="k", label="90% active")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("Fraction active (Fact)")
    axes[1].set_title("GLP1R activation during Cavalitide treatment")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
