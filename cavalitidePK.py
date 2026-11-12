# cavalitidePK.py
# Adapted from semaglutidePK.py: explicit Euler's method replaced by scipy.integrate.solve_ivp,
# and Cavalitide (daily pill) added.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===== choose product: 'Ozempic', 'Wegovy' or 'Cavalitide' =====
#drug = 'Ozempic'
#drug = 'Wegovy'
drug = 'Cavalitide'

# product-specific parameters
products = {
    'Ozempic':    dict(Favailable=0.89,   # bioavailability
                       ka=0.029,          # 1/h, absorption rate
                       D=2.0,             # mg per dose
                       interval=168,      # h between doses (weekly)
                       n_doses=8,         # number of doses
                       depot='subcutaneous'),
    'Wegovy':     dict(Favailable=0.01,
                       ka=0.0029,
                       D=25,              # mg
                       interval=24,       # h between doses (daily)
                       n_doses=100,
                       depot='gut (oral)'),
    'Cavalitide': dict(Favailable=0.01,
                       ka=0.006,          # increased absorption rate
                       D=21,              # mg; optimized from 25 mg to reach ~0.5 nM free in blood
                       interval=24,       # h between doses (daily)
                       n_doses=100,
                       depot='gut (oral)'),
}
p = products[drug]
Favailable, ka, D = p['Favailable'], p['ka'], p['D']
interval, n_doses, depot = p['interval'], p['n_doses'], p['depot']

# shared model parameters
ke = 0.0039     # 1/h, elimination rate
V = 10.8        # L
MW = 4113.6     # g/mol
Ffree = 0.01    # rest of drug bound to albumin

# simulation parameters
dt = 0.1        # h, spacing of the output time points

def ODEfunc(t, y):
    Adepot, Cc = y
    dAdepot = -ka*Adepot            # Adepot: depot amount (mg)
    dCc = ka*Adepot/V - ke*Cc       # Cc: blood concentration (mg/L)
    return [dAdepot, dCc]

def simulate(D, n_doses, interval, dt):
    """Repeated dosing every `interval` h; returns time (h), depot amount (mg),
    and blood concentration (nmol/L)."""
    t_out, Adepot_out, Cc_out = [], [], []
    y = np.array([0.0, 0.0])
    n_steps = int(round(interval / dt))      # output points per dosing interval
    for i in range(n_doses):
        y[0] += Favailable*D                 # absorbed fraction of dose enters the depot
        t0 = i * interval
        t_seg = t0 + dt * np.arange(n_steps + 1)

        # built-in solver (replaces the explicit Euler loop)
        sol = solve_ivp(ODEfunc, (t_seg[0], t_seg[-1]), y, t_eval=t_seg, rtol=1e-8, atol=1e-10)

        t_out.append(sol.t)
        Adepot_out.append(sol.y[0])
        Cc_out.append(sol.y[1])
        y = sol.y[:, -1].copy()              # carry state into next interval
    t = np.concatenate(t_out)
    Adepot_out = np.concatenate(Adepot_out)
    Cc_out = np.concatenate(Cc_out)
    Cc_nM = Cc_out * 1e-3 / MW * 1e9
    return t, Adepot_out, Cc_nM

if __name__ == "__main__":
    # --- run ---
    t, Adepot, Cc = simulate(D, n_doses, interval, dt)

    last = t >= t[-1] - interval
    print(f"{drug}: steady-state blood concentration = {Cc[last].mean():.1f} nM, "
          f"free in blood = {Ffree*Cc[last].mean():.3f} nM")

    # --- plot ---
    plt.plot(t / 24, Adepot, label=f"{depot} depot [mg]")
    plt.plot(t / 24, Ffree*Cc, label="free in blood [nM]")
    plt.xlabel("Time (days)")
    plt.ylabel(drug if drug == 'Cavalitide' else "Semaglutide")
    plt.title(f"{drug}: {D} mg every {interval} h, {n_doses} doses")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
