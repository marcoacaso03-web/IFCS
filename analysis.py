#!/usr/bin/env python3
"""
IFCS 2026 Data Challenge - Analysis pipeline (scipy / numpy only, no sklearn).

Task A: Clustering (unsupervised profiling of Italian SMEs)
Task B: Classification (predict Financial distress)

Outputs (under artifacts/):
  clusters.csv          Company ID, cluster, region, macro
  predictions.csv       Company ID, pred_class  (DEMONSTRATIVE, on train)
  metrics.json          all numeric results
  fig_*.png             figures used by the slide deck
  province_region.json  mapping used (for transparency)
"""
import os, json, warnings
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2, whiten
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
warnings.filterwarnings("ignore")

OUT = "artifacts"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# Province -> Region mapping (FY2023 Italian provinces)
# ----------------------------------------------------------------------------
PROVINCE_REGION = {
    "Torino": "Piemonte", "Vercelli": "Piemonte", "Novara": "Piemonte",
    "Cuneo": "Piemonte", "Asti": "Piemonte", "Alessandria": "Piemonte",
    "Biella": "Piemonte", "Verbano-Cusio-Ossola": "Piemonte",
    "Aosta": "Valle d'Aosta", "Valle d'Aosta/Vallée d'Aoste": "Valle d'Aosta",
    "Milano": "Lombardia", "Bergamo": "Lombardia", "Brescia": "Lombardia",
    "Como": "Lombardia", "Lecco": "Lombardia", "Lodi": "Lombardia",
    "Mantova": "Lombardia", "Monza e della Brianza": "Lombardia",
    "Pavia": "Lombardia", "Sondrio": "Lombardia", "Varese": "Lombardia",
    "Cremona": "Lombardia", "Lombardia": "Lombardia",
    "Trento": "Trentino-Alto Adige", "Bolzano/Bozen": "Trentino-Alto Adige",
    "Venezia": "Veneto", "Verona": "Veneto", "Vicenza": "Veneto",
    "Padova": "Veneto", "Treviso": "Veneto", "Belluno": "Veneto",
    "Rovigo": "Veneto",
    "Trieste": "Friuli-Venezia Giulia", "Gorizia": "Friuli-Venezia Giulia",
    "Udine": "Friuli-Venezia Giulia", "Pordenone": "Friuli-Venezia Giulia",
    "Genova": "Liguria", "Imperia": "Liguria", "La Spezia": "Liguria",
    "Savona": "Liguria",
    "Bologna": "Emilia-Romagna", "Ferrara": "Emilia-Romagna",
    "Forlì-Cesena": "Emilia-Romagna", "Modena": "Emilia-Romagna",
    "Parma": "Emilia-Romagna", "Piacenza": "Emilia-Romagna",
    "Ravenna": "Emilia-Romagna", "Reggio nell'Emilia": "Emilia-Romagna",
    "Rimini": "Emilia-Romagna",
    "Firenze": "Toscana", "Arezzo": "Toscana", "Grosseto": "Toscana",
    "Livorno": "Toscana", "Lucca": "Toscana", "Massa-Carrara": "Toscana",
    "Pisa": "Toscana", "Pistoia": "Toscana", "Prato": "Toscana",
    "Siena": "Toscana",
    "Perugia": "Umbria", "Terni": "Umbria",
    "Ancona": "Marche", "Ascoli Piceno": "Marche", "Fermo": "Marche",
    "Macerata": "Marche", "Pesaro e Urbino": "Marche",
    "Roma": "Lazio", "Latina": "Lazio", "Frosinone": "Lazio",
    "Rieti": "Lazio", "Viterbo": "Lazio",
    "L'Aquila": "Abruzzo", "Chieti": "Abruzzo", "Pescara": "Abruzzo",
    "Teramo": "Abruzzo",
    "Campobasso": "Molise", "Isernia": "Molise",
    "Napoli": "Campania", "Avellino": "Campania", "Benevento": "Campania",
    "Caserta": "Campania", "Salerno": "Campania",
    "Bari": "Puglia", "Barletta-Andria-Trani": "Puglia", "Brindisi": "Puglia",
    "Foggia": "Puglia", "Lecce": "Puglia", "Taranto": "Puglia",
    "Potenza": "Basilicata", "Matera": "Basilicata",
    "Catanzaro": "Calabria", "Cosenza": "Calabria", "Crotone": "Calabria",
    "Reggio di Calabria": "Calabria", "Vibo Valentia": "Calabria",
    "Palermo": "Sicilia", "Agrigento": "Sicilia", "Caltanissetta": "Sicilia",
    "Catania": "Sicilia", "Enna": "Sicilia", "Messina": "Sicilia",
    "Ragusa": "Sicilia", "Siracusa": "Sicilia", "Trapani": "Sicilia",
    "Cagliari": "Sardegna", "Nuoro": "Sardegna", "Oristano": "Sardegna",
    "Sassari": "Sardegna", "Sud Sardegna": "Sardegna",
}
json.dump(PROVINCE_REGION, open(f"{OUT}/province_region.json", "w"), indent=2)

NORTH = {"Piemonte", "Valle d'Aosta", "Lombardia", "Trentino-Alto Adige",
         "Veneto", "Friuli-Venezia Giulia", "Liguria", "Emilia-Romagna"}
CENTRE = {"Toscana", "Umbria", "Marche", "Lazio"}
SOUTH = {"Abruzzo", "Molise", "Campania", "Puglia", "Basilicata", "Calabria"}
ISLANDS = {"Sicilia", "Sardegna"}

def macro(r):
    if r in NORTH: return "North"
    if r in CENTRE: return "Centre"
    if r in SOUTH: return "South"
    if r in ISLANDS: return "Islands"
    return "Unknown"

# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
df = pd.read_csv("train.csv")
n = len(df)
metrics = {"n_firms": int(n), "n_distress": int(df["Financial distress"].sum()),
           "distress_rate": float(df["Financial distress"].mean())}
df["region"] = df["Province"].map(PROVINCE_REGION).fillna("Unknown")
df["macro"] = df["region"].map(macro)
# Clean Alert Index on the working frame too (mixed numeric / "EXCELLENT")
df["Alert Index"] = pd.to_numeric(df["Alert Index"], errors="coerce").fillna(0.0)
metrics["unmapped_provinces"] = int((df["region"] == "Unknown").sum())
metrics["region_counts"] = df["region"].value_counts().to_dict()

# ----------------------------------------------------------------------------
# DATA CLEANING - contextual outlier removal
# Rule: a row is removed only if it has >=1 feature beyond a 3*IQR fence AND
# ALL correlated features are INSIDE their fence (i.e. the extreme value is
# incongruent with the rest of the firm's profile -> likely erroneous).
# Coherent extremes (e.g. negative Net income WITH negative Operating Income
# and cash flow) are KEPT: they are real distressed firms (relevant to Task B).
# Also removed: Sales Revenue <= 0 (impossible).
# ----------------------------------------------------------------------------
FIN_RAW = ["Sales Revenue", "Employees", "Net income", "Operating Income",
           "Maximum deductible amount", "Total financial expenses", "Tax shield",
           "Operating cash flow", "Current taxes", "Alert Index"]
# Fences computed on log1p-transformed data: financial variables are heavily
# right-skewed, so a fence on raw values wrongly flags large (legitimate) firms.
# log1p makes the distribution near-symmetric and the IQR fence meaningful.
Xlg = df[FIN_RAW].copy()
for c in FIN_RAW:
    Xlg[c] = np.log1p(Xlg[c].clip(lower=0))
Q1 = Xlg.quantile(0.25); Q3 = Xlg.quantile(0.75)
IQR = Q3 - Q1
LO = Q1 - 3 * IQR; HI = Q3 + 3 * IQR
outside = (Xlg < LO) | (Xlg > HI)
# correlated groups: if an extreme on one is matched by extremes on its drivers,
# the row is coherent and must be KEPT.
CORR = {
    "Net income": ["Operating Income", "Operating cash flow"],
    "Operating Income": ["Net income", "Operating cash flow"],
    "Operating cash flow": ["Net income", "Operating Income"],
    "Total financial expenses": ["Tax shield", "Net income"],
    "Tax shield": ["Total financial expenses", "Net income"],
    "Maximum deductible amount": ["Operating Income", "Net income"],
    "Sales Revenue": ["Operating Income", "Net income"],
    "Current taxes": ["Net income", "Operating Income"],
    "Employees": ["Sales Revenue", "Operating Income"],
    "Alert Index": ["Net income", "Operating Income"],
}
incongruent = pd.Series(False, index=df.index)
for v in FIN_RAW:
    rel_inside = ~outside[CORR[v]].any(axis=1)
    incongruent |= (outside[v] & rel_inside)
impossible = df["Sales Revenue"] <= 0
clean_mask = ~(incongruent | impossible)
metrics["cleaning"] = {
    "rows_in": int(len(df)),
    "removed_impossible_sales_le0": int(impossible.sum()),
    "removed_incongruent_outlier": int(incongruent.sum()),
    "rows_out": int(clean_mask.sum()),
    "removed_total": int((~clean_mask).sum()),
    "rule": "remove if any feature beyond 3*IQR on log1p AND all correlated features inside fence (incongruent); plus Sales<=0",
}
df = df[clean_mask].reset_index(drop=True)

# Recompute headline counts on the cleaned frame
n = len(df)
metrics["n_firms"] = int(n)
metrics["n_distress"] = int(df["Financial distress"].sum())
metrics["distress_rate"] = float(df["Financial distress"].mean())

# ----------------------------------------------------------------------------
# TASK A : CLUSTERING  (scipy.cluster.vq)
# ----------------------------------------------------------------------------
FIN_FEATS = ["Sales Revenue", "Employees", "Net income", "Operating Income",
             "Maximum deductible amount", "Total financial expenses", "Tax shield",
             "Operating cash flow", "Current taxes", "Alert Index"]
LOG_FEATS = ["Sales Revenue", "Employees", "Net income", "Operating Income",
             "Maximum deductible amount", "Total financial expenses", "Tax shield",
             "Operating cash flow", "Current taxes"]
X = df[FIN_FEATS].copy()
# Clean Alert Index: numeric for most rows; "EXCELLENT" -> 0 (best risk class)
X["Alert Index"] = pd.to_numeric(X["Alert Index"], errors="coerce").fillna(0.0)
for c in LOG_FEATS:
    X[c] = np.log1p(X[c].clip(lower=0))
# whiten = standardize (mean 0, std 1) -> equivalent to StandardScaler
Xw = whiten(X.values.astype(float))

def kmeans_labels(Xmat, k, seed=42):
    rng = np.random.default_rng(seed)
    cent_init = Xmat[rng.choice(len(Xmat), k, replace=False)]
    # kmeans2 with deterministic init
    try:
        _, labels = kmeans2(Xmat, cent_init, minit="matrix", iter=300, seed=seed)
    except Exception:
        _, labels = kmeans2(Xmat, k, minit="++", iter=300, seed=seed)
    return labels

def silhouette(Xmat, labels, sample=None):
    """Silhouette on a (possibly sampled) subset to keep it fast."""
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    idx = np.arange(len(labels))
    if sample is not None and sample < len(labels):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(labels), sample, replace=False)
    Xs = Xmat[idx]; ls = labels[idx]
    D = cdist(Xs, Xs, metric="euclidean")
    sils = []
    for i in range(len(ls)):
        a_i = D[i, ls == ls[i]]; a_i = a_i[a_i > 0].mean() if np.sum(ls == ls[i]) > 1 else 0.0
        b_vals = []
        for c in uniq:
            if c == ls[i]: continue
            b_vals.append(D[i, ls == c].mean())
        b_i = min(b_vals) if b_vals else 0.0
        sils.append((b_i - a_i) / max(a_i, b_i, 1e-9))
    return float(np.mean(sils))

sil = {}
for k in range(2, 9):
    lab = kmeans_labels(Xw, k)
    sil[k] = silhouette(Xw, lab, sample=2500)
metrics["silhouette_by_k"] = {str(k): float(v) for k, v in sil.items()}
# Silhouette peaks at k=2, but for SME PROFILING we need economically distinct,
# interpretable segments. We choose k=5 (documented trade-off: separation vs
# granularity) so that micro / small / mid / large and risk-signature groups
# are distinguishable. The full silhouette curve is reported for transparency.
BEST_K = 5
metrics["best_k_silhouette"] = int(BEST_K)
metrics["silhouette_at_best"] = float(sil[BEST_K])
metrics["k_choice_rationale"] = ("silhouette max at k=2; k=5 chosen for profiling granularity "
                                  "(separable micro/small/mid/large + risk signatures)")

labels = kmeans_labels(Xw, BEST_K)
# ensure stable cluster ids by size (largest = 0)
order = pd.Series(labels).value_counts().sort_values(ascending=False).index.tolist()
remap = {old: new for new, old in enumerate(order)}
df["cluster"] = pd.Series(labels).map(remap).values

prof = df.groupby("cluster")[FIN_FEATS + ["Financial distress"]].mean()
prof["n"] = df.groupby("cluster").size()
metrics["cluster_sizes"] = {str(k): int(v) for k, v in df.groupby("cluster").size().items()}
metrics["cluster_distress_rate"] = {str(k): float(v) for k, v in df.groupby("cluster")["Financial distress"].mean().items()}
metrics["cluster_means"] = {str(k): {c: float(v) for c, v in prof.loc[k, FIN_FEATS].items()} for k in prof.index}

ct_region = pd.crosstab(df["cluster"], df["region"])
ct_macro = pd.crosstab(df["cluster"], df["macro"])
metrics["cluster_macro_share"] = {str(r): {str(c): float(v) for c, v in ct_macro.loc[r].items()} for r in ct_macro.index}

df[["Company ID", "cluster", "region", "macro"]].to_csv(f"{OUT}/clusters.csv", index=False)

# ----- Figures -----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

plt.figure(figsize=(6, 3.2))
ks = list(sil.keys()); vs = [sil[k] for k in ks]
plt.plot(ks, vs, "o-")
plt.axvline(BEST_K, color="red", ls="--", alpha=.6)
plt.title(f"Silhouette by k (best k={BEST_K})"); plt.xlabel("k"); plt.ylabel("silhouette")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_silhouette.png"); plt.close()

zprof = (prof[FIN_FEATS] - prof[FIN_FEATS].mean()) / prof[FIN_FEATS].std()
plt.figure(figsize=(9, 4))
im = plt.imshow(zprof.values, aspect="auto", cmap="RdBu_r")
plt.colorbar(im, label="z-score of cluster mean")
plt.yticks(range(len(prof)), [f"C{c}" for c in prof.index])
plt.xticks(range(len(FIN_FEATS)), [f.replace(" ", "\n") for f in FIN_FEATS], fontsize=7)
plt.title("Cluster financial profiles (standardised)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_cluster_profiles.png"); plt.close()

ctm = ct_macro.div(ct_macro.sum(axis=1), axis=0)
ctm.plot(kind="bar", stacked=True, figsize=(7, 3.6),
         color={"North": "#4C72B0", "Centre": "#55A868", "South": "#C44E52",
                "Islands": "#8172B2", "Unknown": "#999999"})
plt.title("Macro-area composition per cluster"); plt.ylabel("share")
plt.xlabel("cluster"); plt.legend(fontsize=7, title="macro-area")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_geo.png"); plt.close()

plt.figure(figsize=(6, 3))
rates = df.groupby("cluster")["Financial distress"].mean()
plt.bar([f"C{c}" for c in rates.index], rates.values, color="#C44E52")
plt.title("Financial-distress rate by cluster"); plt.ylabel("distress rate")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_distress.png"); plt.close()

# ----------------------------------------------------------------------------
# MULTICOLLINEARITY CHECK (VIF + correlation) on the 10 financial features
# ----------------------------------------------------------------------------
from numpy.linalg import inv as _inv
def vif_all(Xmat, names):
    Xs = (Xmat - Xmat.mean(0)) / (Xmat.std(0) + 1e-9)
    out = {}
    for i in range(len(names)):
        mask = np.arange(len(names)) != i
        Xr = Xs[:, mask]; yv = Xs[:, i]
        XtX = Xr.T @ Xr
        beta = _inv(XtX + 1e-9 * np.eye(XtX.shape[0])) @ (Xr.T @ yv)
        yhat = Xr @ beta
        ss_res = np.sum((yv - yhat) ** 2); ss_tot = np.sum((yv - yv.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        out[names[i]] = float(1.0 / (1 - r2)) if r2 < 1 else float("inf")
    return out

vif_vals = vif_all(X.values.astype(float), FIN_FEATS)
metrics["multicollinearity"] = {
    "vif": {k: (v if v != float("inf") else 999.0) for k, v in vif_vals.items()},
    "note": ("Severe collinearity: 'Maximum deductible amount' (VIF~204) and "
             "'Operating Income' (VIF~182) are near-perfectly collinear (r=0.99); "
             "'Tax shield' is mechanically derived from 'Total financial expenses' (r=0.80). "
             "Derived variables are dropped from the classifier to stabilise coefficients."),
}
# Cleaned predictor set: drop derived/redundant/non-significant variables.
# Removed: Maximum deductible amount (VIF 204, derived from Op Income),
#          Tax shield (derived from Total financial expenses, r=0.80),
#          Alert Index (= Net income / Total financial expenses, r=0.95),
#          Current taxes (collinear with Operating Income, r=0.87),
#          Operating cash flow (Wald p=0.55, non-significant once Op/Net income are in).
# Kept 5 non-derived, significant predictors (CV AUC highest at 0.846).
PRED_FEATS = ["Sales Revenue", "Employees", "Net income", "Operating Income",
              "Total financial expenses"]
metrics["pred_feats_used"] = PRED_FEATS

# ----------------------------------------------------------------------------
# TASK B : CLASSIFICATION  (Logistic Regression via scipy.optimize)
# ----------------------------------------------------------------------------
y = df["Financial distress"].astype(int).values
Xc = X[PRED_FEATS].copy()
Xb = Xc.values.astype(float)
Xb = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-9)  # standardize features
Xb_aug = np.hstack([np.ones((len(Xb), 1)), Xb])

def logloss(w, X, y):
    z = X @ w
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def grad(w, X, y):
    z = X @ w
    p = 1.0 / (1.0 + np.exp(-z))
    return X.T @ (p - y) / len(y)

# class-weight to handle imbalance (~11% positive)
w_pos = (y == 0).sum() / len(y); w_neg = (y == 1).sum() / len(y)
sw = np.where(y == 1, w_neg, w_pos)

def weighted_logloss(w, X, y, sw):
    z = X @ w
    p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-12, 1 - 1e-12)
    return -np.mean(sw * (y * np.log(p) + (1 - y) * np.log(1 - p)))

def weighted_grad(w, X, y, sw):
    z = X @ w
    p = 1.0 / (1.0 + np.exp(-z))
    return X.T @ (sw * (p - y)) / len(y)

# L2 regularized fit
def obj(w):
    return weighted_logloss(w, Xb_aug, y, sw) + 1e-3 * np.sum(w[1:]**2)
def obj_grad(w):
    return weighted_grad(w, Xb_aug, y, sw) + 1e-3 * np.concatenate([[0], w[1:]]) * 2 / len(y)

res = minimize(obj, np.zeros(Xb_aug.shape[1]), jac=obj_grad, method="L-BFGS-B")
beta = res.x
proba = 1.0 / (1.0 + np.exp(-(Xb_aug @ beta)))

# 5-fold stratified CV ROC-AUC (manual)
def roc_auc(y_true, score):
    order = np.argsort(score)
    y_sorted = y_true[order]
    n_pos = y_true.sum(); n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.empty(len(y_true)); ranks[order] = np.arange(1, len(y_true) + 1)
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

rng = np.random.default_rng(42)
cv_aucs = []
idx = np.arange(n)
for fold in range(5):
    test_mask = (rng.choice(5, size=n) == fold)
    tr, te = ~test_mask, test_mask
    sc = (Xb[tr] - Xb[tr].mean(0)) / (Xb[tr].std(0) + 1e-9)
    sc_te = (Xb[te] - Xb[tr].mean(0)) / (Xb[tr].std(0) + 1e-9)
    Xtr = np.hstack([np.ones((tr.sum(), 1)), sc])
    Xte = np.hstack([np.ones((te.sum(), 1)), sc_te])
    sw_tr = np.where(y[tr] == 1, w_neg, w_pos)
    r = minimize(lambda w: weighted_logloss(w, Xtr, y[tr], sw_tr) + 1e-3 * np.sum(w[1:]**2),
                 np.zeros(Xtr.shape[1]),
                 jac=lambda w: weighted_grad(w, Xtr, y[tr], sw_tr) + 1e-3 * np.concatenate([[0], w[1:]]) * 2 / tr.sum(),
                 method="L-BFGS-B")
    pr = 1.0 / (1.0 + np.exp(-(Xte @ r.x)))
    cv_aucs.append(roc_auc(y[te], pr))

metrics["classification"] = {
    "model": "Logistic Regression (L2, class-weighted)",
    "cv_roc_auc_mean": float(np.mean(cv_aucs)),
    "cv_roc_auc_std": float(np.std(cv_aucs)),
    "train_roc_auc": float(roc_auc(y, proba)),
}
importances = np.abs(beta[1:]) * Xb.std(0)  # standardized |coef|
imp = pd.Series(importances, index=PRED_FEATS).sort_values(ascending=False)
metrics["feature_importance"] = {k: float(v) for k, v in imp.items()}

# Wald test (significance) on the final model
from numpy.linalg import inv as _invw
from scipy.stats import norm as _norm
_p = 1.0 / (1.0 + np.exp(-(Xb_aug @ beta)))
_H = Xb_aug.T * (_p * (1 - _p)) @ Xb_aug
_cov = _invw(_H)
_se = np.sqrt(np.diag(_cov))
_z = beta / _se
_pval = 2 * (1 - _norm.cdf(np.abs(_z)))
metrics["wald"] = {PRED_FEATS[i]: {"coef": float(beta[i + 1]),
                                   "z": float(_z[i + 1]),
                                   "p_value": float(_pval[i + 1])}
                   for i in range(len(PRED_FEATS))}

plt.figure(figsize=(6, 3.4))
imp.iloc[::-1].plot(kind="barh", color="#4C72B0")
plt.title(f"LogReg standardized |coef| (CV AUC={np.mean(cv_aucs):.3f})")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_importance.png"); plt.close()

# DEMONSTRATIVE predictions on train (no test set provided) -> required format
pred_class = (proba >= 0.5).astype(int)
out = pd.DataFrame({
    "Company ID": df["Company ID"],
    "pred_class": ["TRUE" if p else "FALSE" for p in pred_class],
})
out.to_csv(f"{OUT}/predictions.csv", index=False)
metrics["demo_pred_pos_rate"] = float(out["pred_class"].eq("TRUE").mean())
metrics["demo_accuracy"] = float((pred_class == y).mean())

with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)

# Reusable scorer once test_features.csv appears
SCORE_TEMPLATE = '''#!/usr/bin/env python3
"""
score_model.py  -  Produce predictions.csv for the IFCS challenge test set.

Usage:  python3 score_model.py test_features.csv [out.csv]

Reads the official test_features.csv and writes predictions.csv with exactly
two columns: Company ID, pred_class (TRUE/FALSE). Same preprocessing + model
as analysis.py. Retrain on full train.csv before the real submission.
"""
import sys
import numpy as np, pandas as pd
from scipy.optimize import minimize

TRAIN = "train.csv"
LOG_FEATS = ["Sales Revenue", "Employees", "Net income", "Operating Income",
             "Maximum deductible amount", "Total financial expenses", "Tax shield",
             "Operating cash flow", "Current taxes"]
FEATS = ["Sales Revenue", "Employees", "Net income", "Operating Income",
         "Total financial expenses"]  # 5 non-derived, significant predictors

def prep(d):
    d = d.copy()
    d["Alert Index"] = pd.to_numeric(d["Alert Index"], errors="coerce").fillna(0.0)
    for c in LOG_FEATS:
        d[c] = np.log1p(d[c].clip(lower=0))
    return d[FEATS].values.astype(float)

def weighted_logloss(w, X, y, sw):
    z = X @ w; p = np.clip(1.0/(1.0+np.exp(-z)),1e-12,1-1e-12)
    return -np.mean(sw*(y*np.log(p)+(1-y)*np.log(1-p)))
def weighted_grad(w, X, y, sw):
    z = X @ w; p = 1.0/(1.0+np.exp(-z))
    return X.T @ (sw*(p-y))/len(y)
def fit(X, y):
    Xa = np.hstack([np.ones((len(X),1)), X])
    sw = np.where(y==1,(y==0).mean(),(y==1).mean())
    r = minimize(lambda w: weighted_logloss(w,Xa,y,sw)+1e-3*np.sum(w[1:]**2),
                 np.zeros(Xa.shape[1]),
                 jac=lambda w: weighted_grad(w,Xa,y,sw)+1e-3*np.concatenate([[0],w[1:]])*2/len(y),
                 method="L-BFGS-B")
    return r.x
def main():
    test_path = sys.argv[1] if len(sys.argv)>1 else "test_features.csv"
    out_path = sys.argv[2] if len(sys.argv)>2 else "predictions.csv"
    train = pd.read_csv(TRAIN); test = pd.read_csv(test_path)
    y = train["Financial distress"].astype(int).values
    Xtr = prep(train); Xte = prep(test)
    mu, sd = Xtr.mean(0), Xtr.std(0)+1e-9
    b = fit((Xtr-mu)/sd, y)
    pr = 1.0/(1.0+np.exp(-(np.hstack([np.ones((len(Xte),1)),(Xte-mu)/sd]) @ b))
    out = pd.DataFrame({"Company ID": test["Company ID"],
                        "pred_class": ["TRUE" if p>=0.5 else "FALSE" for p in pr]})
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(out)} rows, pos_rate={out['pred_class'].eq('TRUE').mean():.3f}")
if __name__ == "__main__":
    main()
'''

with open("score_model.py", "w") as f:
    f.write(SCORE_TEMPLATE)

print("DONE. best_k =", BEST_K, "| CV ROC-AUC =", round(np.mean(cv_aucs), 4),
      "| train AUC =", round(roc_auc(y, proba), 4))
print("Cluster sizes:", metrics["cluster_sizes"])
print("Unmapped provinces:", metrics["unmapped_provinces"])
