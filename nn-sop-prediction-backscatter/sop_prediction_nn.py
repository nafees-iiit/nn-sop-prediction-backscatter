"""
Neural-Network Surrogate for Secrecy Outage Probability (SOP) Prediction
=======================================================================
Author : Mohammad Nafees (Ph.D. Scholar, ECE, MNNIT Allahabad)

Problem
-------
In ambient-backscatter IoT with optimal tag selection (OTS), the secrecy outage
probability depends on many system parameters: transmit SNR, Nakagami-m fading,
number of tags K, eavesdropper channel strength, target secrecy rate R_s and the
reflection coefficient beta. Computing SOP needs either long closed-form
expressions or thousands of Monte Carlo trials for EVERY parameter setting.

Idea
----
Generate a dataset with the Monte Carlo simulator, then train a neural network
(regression) that maps the 6 system parameters -> SOP. Once trained, the network
predicts SOP for new, unseen settings almost instantly. This is called a
"surrogate model" (a fast stand-in for an expensive simulator).
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# ------------------------------------------------------------------
# 1. Settings
# ------------------------------------------------------------------
SEED       = 42
N_SAMPLES  = 4000        # number of parameter settings in the dataset
N_TRIALS   = 10000       # Monte Carlo trials used to label each setting
SOP_FLOOR  = 1 / N_TRIALS  # smallest SOP we can measure with N_TRIALS
OMEGA_MAIN = 1.0         # average gain of S->tag and tag->reader links
rng = np.random.default_rng(SEED)

# Parameter ranges (the "design space" the network must learn)
RANGES = {
    "snr_db": (0.0, 30.0),   # transmit SNR in dB
    "m":      (1.0, 4.0),    # Nakagami-m fading parameter
    "K":      (1, 8),        # number of tags (integer)
    "omega_e": (0.1, 1.0),   # average gain of tag->eavesdropper link
    "rs":     (0.2, 2.0),    # target secrecy rate (bits/s/Hz)
    "beta":   (0.2, 0.9),    # reflection coefficient
}
FEATURES = list(RANGES.keys())


# ------------------------------------------------------------------
# 2. Monte Carlo simulator (the slow "ground truth")
# ------------------------------------------------------------------
def simulate_sop(snr_db, m, K, omega_e, rs, beta, n_trials=N_TRIALS, gen=rng):
    """SOP of optimal tag selection (OTS) by Monte Carlo simulation."""
    K = int(K)
    snr = 10 ** (snr_db / 10)
    h = gen.gamma(m, OMEGA_MAIN / m, size=(n_trials, K))   # |S->tag|^2
    g = gen.gamma(m, OMEGA_MAIN / m, size=(n_trials, K))   # |tag->reader|^2
    f = gen.gamma(m, omega_e / m, size=(n_trials, K))      # |tag->eve|^2
    cs = np.log2(1 + snr * beta * h * g) - np.log2(1 + snr * beta * h * f)
    cs_best = np.maximum(cs.max(axis=1), 0.0)              # OTS: best tag
    return np.mean(cs_best < rs)


# ------------------------------------------------------------------
# 3. Build the dataset: random parameter settings -> simulated SOP
# ------------------------------------------------------------------
def random_settings(n):
    df = pd.DataFrame({
        "snr_db": rng.uniform(*RANGES["snr_db"], n),
        "m": rng.uniform(*RANGES["m"], n),
        "K": rng.integers(RANGES["K"][0], RANGES["K"][1] + 1, n),
        "omega_e": rng.uniform(*RANGES["omega_e"], n),
        "rs": rng.uniform(*RANGES["rs"], n),
        "beta": rng.uniform(*RANGES["beta"], n),
    })
    return df

t0 = time.time()
data = random_settings(N_SAMPLES)
data["sop"] = [simulate_sop(**row) for row in data[FEATURES].to_dict("records")]
sim_time = time.time() - t0
data.to_csv("results/dataset.csv", index=False)
print(f"Dataset: {N_SAMPLES} settings x {N_TRIALS} Monte Carlo trials "
      f"generated in {sim_time:.1f} s ({1000*sim_time/N_SAMPLES:.2f} ms per setting)")

# SOP ranges from ~1e-4 to 1, so we learn log10(SOP) instead of SOP itself
y = np.log10(np.clip(data["sop"].values, SOP_FLOOR, 1.0))
X = data[FEATURES].values

# 70 % train, 15 % validation, 15 % test
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)
print(f"Train = {len(y_tr)}, Validation = {len(y_val)}, Test = {len(y_te)}")

# ------------------------------------------------------------------
# 4. Three regression models
# ------------------------------------------------------------------
models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                           n_jobs=-1, random_state=SEED),
    "Neural Network (MLP)": make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 64, 32), activation="relu",
                     learning_rate_init=1e-3, max_iter=2000, early_stopping=True,
                     validation_fraction=0.15, n_iter_no_change=50,
                     random_state=SEED)),
}

rows = []
for name, model in models.items():
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_t = time.time() - t0
    pred_val, pred_te = model.predict(X_val), model.predict(X_te)
    # error in the SOP domain, only where SOP is measurable (> 1e-3)
    sop_true, sop_pred = 10 ** y_te, 10 ** np.clip(pred_te, np.log10(SOP_FLOOR), 0)
    ok = sop_true > 1e-3
    rel_err = np.median(np.abs(sop_pred[ok] - sop_true[ok]) / sop_true[ok])
    rows.append({"Model": name,
                 "Val R2 (log10 SOP)": r2_score(y_val, pred_val),
                 "Test R2 (log10 SOP)": r2_score(y_te, pred_te),
                 "Test MAE (log10 SOP)": mean_absolute_error(y_te, pred_te),
                 "Median relative error (SOP > 1e-3)": rel_err,
                 "Training time (s)": fit_t})
metrics = pd.DataFrame(rows)
metrics.to_csv("results/model_comparison.csv", index=False)
print("\nModel comparison on unseen TEST data:")
print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

nn = models["Neural Network (MLP)"]
mlp = nn.named_steps["mlpregressor"]
print(f"\nNeural network stopped after {mlp.n_iter_} epochs "
      f"(best validation score = {mlp.best_validation_score_:.4f})")

# ------------------------------------------------------------------
# 5. Speed: neural network vs Monte Carlo simulation
# ------------------------------------------------------------------
probe = random_settings(2000)
t0 = time.time(); nn.predict(probe[FEATURES].values); t_nn = (time.time() - t0) / 2000
t0 = time.time()
for r in probe[FEATURES].head(200).to_dict("records"):
    simulate_sop(**r, n_trials=100000)
t_mc = (time.time() - t0) / 200
print(f"\nTime per SOP value: Monte Carlo (100k trials) = {1000*t_mc:.1f} ms, "
      f"Neural network = {1e6*t_nn:.1f} us  ->  about {t_mc/t_nn:,.0f}x faster")

# ------------------------------------------------------------------
# 6. Figures
# ------------------------------------------------------------------
# (a) Parity plot: predicted vs true on the test set
pred_te = nn.predict(X_te)
plt.figure(figsize=(5.5, 5.2))
plt.scatter(y_te, pred_te, s=8, alpha=0.5, color="tab:red")
lims = [np.log10(SOP_FLOOR) - 0.2, 0.2]
plt.plot(lims, lims, "k--", lw=1, label="perfect prediction")
plt.xlim(lims); plt.ylim(lims)
plt.xlabel("True log10(SOP)  [Monte Carlo]"); plt.ylabel("Predicted log10(SOP)  [Neural network]")
plt.title(f"Parity plot on unseen test data (R² = {r2_score(y_te, pred_te):.3f})")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig("results/parity_plot.png", dpi=200); plt.close()

# (b) SOP vs SNR curves for brand-new settings: NN vs fresh Monte Carlo
cases = [dict(m=1.0, K=1, omega_e=0.5, rs=0.5, beta=0.5),
         dict(m=2.0, K=4, omega_e=0.5, rs=0.5, beta=0.5),
         dict(m=3.0, K=8, omega_e=0.3, rs=1.0, beta=0.7)]
snr_grid = np.arange(0, 31, 2)
gen_check = np.random.default_rng(2026)          # independent random stream
plt.figure(figsize=(7.2, 5))
curve_rows = []
for c, col in zip(cases, ["tab:blue", "tab:green", "tab:purple"]):
    mc = [simulate_sop(s, **c, n_trials=200000, gen=gen_check) for s in snr_grid]
    Xc = np.array([[s, c["m"], c["K"], c["omega_e"], c["rs"], c["beta"]] for s in snr_grid])
    pr = 10 ** np.clip(nn.predict(Xc), np.log10(SOP_FLOOR), 0)
    lab = f"m={c['m']:.0f}, K={c['K']}, Ω_E={c['omega_e']}, R_s={c['rs']}, β={c['beta']}"
    plt.semilogy(snr_grid, np.maximum(mc, 1e-5), "o", color=col, label=f"Monte Carlo: {lab}")
    plt.semilogy(snr_grid, pr, "-", color=col, label="Neural network")
    for s, a, b in zip(snr_grid, mc, pr):
        curve_rows.append({**c, "snr_db": s, "sop_monte_carlo": a, "sop_nn": b})
pd.DataFrame(curve_rows).to_csv("results/sop_curves_nn_vs_mc.csv", index=False)
plt.xlabel("Transmit SNR (dB)"); plt.ylabel("Secrecy Outage Probability")
plt.title("Unseen settings: neural network vs fresh Monte Carlo (200k trials)")
plt.grid(True, which="both", alpha=0.3); plt.legend(fontsize=7)
plt.tight_layout(); plt.savefig("results/sop_vs_snr_nn_vs_mc.png", dpi=200); plt.close()

# (c) Training curve of the neural network
plt.figure(figsize=(6.5, 4.2))
plt.semilogy(mlp.loss_curve_, label="training loss")
plt.xlabel("Epoch"); plt.ylabel("Loss (squared error)")
plt.title("Neural network training curve"); plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig("results/training_curve.png", dpi=200); plt.close()

# (d) Learning curve: how much data does the network need?
sizes = [200, 400, 800, 1600, len(y_tr)]
r2s = []
for n in sizes:
    model = make_pipeline(StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(64, 64, 32), max_iter=2000,
                                       early_stopping=True, n_iter_no_change=50,
                                       random_state=SEED))
    model.fit(X_tr[:n], y_tr[:n])
    r2s.append(r2_score(y_te, model.predict(X_te)))
pd.DataFrame({"training_samples": sizes, "test_R2": r2s}).to_csv("results/learning_curve.csv", index=False)
plt.figure(figsize=(6.5, 4.2))
plt.plot(sizes, r2s, "o-", color="tab:red")
plt.xlabel("Number of training samples"); plt.ylabel("Test R²")
plt.title("Learning curve: more data -> better surrogate"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/learning_curve.png", dpi=200); plt.close()
print("\nLearning curve (training samples -> test R2): " +
      ", ".join(f"{n}->{r:.3f}" for n, r in zip(sizes, r2s)))
print("\nAll results saved in the 'results' folder.")
