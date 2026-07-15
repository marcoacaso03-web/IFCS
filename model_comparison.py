#!/usr/bin/env python3
"""
model_comparison.py - IFCS Task B model comparison.

Compares three classifiers on the cleaned 4-feature set:
  - Logistic Regression (scipy L-BFGS-B, class-weighted, L2)
  - Random Forest (hand-rolled numpy CART, bootstrap + feature subsampling)
  - Gradient Boosted Trees (hand-rolled numpy additive shallow trees, GBM)

Evaluation: 5-fold stratified CV AND 80/20 stratified holdout.
Metrics: Precision, Recall, F1, F2, ROC-AUC.
F2 weights recall x2 (distress: cost of a false negative is high).

NOTE: sklearn/xgboost are unavailable on this Termux; RF and GBM are numpy
implementations, labelled as such. Logistic Regression uses scipy.
"""
import json, numpy as np, pandas as pd
from scipy.optimize import minimize

OUT = "artifacts"

# ---------------------------------------------------------------------------
# Load + SAME cleaning/feature engineering as analysis.py
# ---------------------------------------------------------------------------
df = pd.read_csv("train.csv")
df["Alert Index"] = pd.to_numeric(df["Alert Index"], errors="coerce").fillna(0.0)
RAW = ["Sales Revenue", "Employees", "Net income", "Operating Income",
       "Maximum deductible amount", "Total financial expenses", "Tax shield",
       "Operating cash flow", "Current taxes", "Alert Index"]
Xlg = df[RAW].copy()
for c in RAW:
    Xlg[c] = np.log1p(Xlg[c].clip(lower=0))
Q1 = Xlg.quantile(0.25); Q3 = Xlg.quantile(0.75); IQR = Q3 - Q1
LO = Q1 - 3 * IQR; HI = Q3 + 3 * IQR
outside = (Xlg < LO) | (Xlg > HI)
CORR = {"Net income": ["Operating Income", "Operating cash flow"],
        "Operating Income": ["Net income", "Operating cash flow"],
        "Operating cash flow": ["Net income", "Operating Income"],
        "Total financial expenses": ["Tax shield", "Net income"],
        "Tax shield": ["Total financial expenses", "Net income"],
        "Maximum deductible amount": ["Operating Income", "Net income"],
        "Sales Revenue": ["Operating Income", "Net income"],
        "Current taxes": ["Net income", "Operating Income"],
        "Employees": ["Sales Revenue", "Operating Income"],
        "Alert Index": ["Net income", "Operating Income"]}
inc = pd.Series(False, index=df.index)
for v in RAW:
    inc |= (outside[v] & ~outside[CORR[v]].any(axis=1))
df = df[~(inc | (df["Sales Revenue"] <= 0))].reset_index(drop=True)
df["Rev_per_Employee"] = df["Sales Revenue"] / df["Employees"].clip(lower=1)

FEATS = ["Operating Income", "Net income", "Total financial expenses", "Rev_per_Employee"]
Xd = df[FEATS].copy()
for c in FEATS:
    Xd[c] = np.log1p(Xd[c].clip(lower=0))
Xs = (Xd - Xd.mean()) / (Xd.std() + 1e-9)
X = Xs.values.astype(float)
y = df["Financial distress"].astype(int).values
n, p = X.shape
rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def prf(y_true, y_hat, thr=0.5):
    yp = (y_hat >= thr).astype(int)
    tp = int(((yp == 1) & (y_true == 1)).sum())
    fp = int(((yp == 1) & (y_true == 0)).sum())
    fn = int(((yp == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0.0
    return prec, rec, f1, f2

def roc_auc(y_true, score):
    order = np.argsort(score); ys = y_true[order]
    a = int(y_true.sum()); b = len(y_true) - a
    if a == 0 or b == 0:
        return float("nan")
    ranks = np.empty(len(y_true)); ranks[order] = np.arange(1, len(y_true) + 1)
    return (ranks[y_true == 1].sum() - a * (a + 1) / 2) / (a * b)

# ---------------------------------------------------------------------------
# Model 1: Logistic Regression (scipy)
# ---------------------------------------------------------------------------
def logreg_fit(Xtr, ytr, Xte):
    sw = np.where(ytr == 1, (ytr == 0).mean(), (ytr == 1).mean())
    Xa = np.hstack([np.ones((len(Xtr), 1)), Xtr])
    Xae = np.hstack([np.ones((len(Xte), 1)), Xte])
    def loss(w):
        z = Xa @ w; p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.mean(sw * (ytr * np.log(p) + (1 - ytr) * np.log(1 - p))) + 1e-3 * np.sum(w[1:]**2)
    def grad(w):
        z = Xa @ w; p = 1 / (1 + np.exp(-z))
        return Xa.T @ (sw * (p - ytr)) / len(ytr) + 1e-3 * np.concatenate([[0], w[1:]]) * 2 / len(ytr)
    r = minimize(loss, np.zeros(Xa.shape[1]), jac=grad, method="L-BFGS-B")
    return 1 / (1 + np.exp(-(Xae @ r.x)))

# ---------------------------------------------------------------------------
# Models 2 & 3: hand-rolled CART-based RF and GBM
# ---------------------------------------------------------------------------
class Node:
    pass

def gini(yy):
    if len(yy) == 0:
        return 0.0
    p1 = yy.mean()
    return 1 - p1**2 - (1 - p1)**2

def variance(yy):
    return yy.var() if len(yy) > 1 else 0.0

def best_split(Xmat, yy, feat_idx, criterion):
    best = None; best_gain = -1
    parent = (gini(yy) if criterion == "gini" else variance(yy))
    n = len(yy)
    for f in feat_idx:
        col = Xmat[:, f]
        thr = np.median(col)
        if np.isnan(thr) or np.unique(col).size < 2:
            continue
        left = col <= thr; right = ~left
        if left.sum() < 2 or right.sum() < 2:
            continue
        gl = (gini(yy[left]) if criterion == "gini" else variance(yy[left]))
        gr = (gini(yy[right]) if criterion == "gini" else variance(yy[right]))
        gain = parent - (left.sum() * gl + right.sum() * gr) / n
        if gain > best_gain:
            best_gain = gain; best = (f, thr, left, right)
    return best

def build_tree(Xmat, yy, depth, max_depth, min_leaf, feat_pool, criterion="gini"):
    imp = (gini(yy) if criterion == "gini" else variance(yy))
    if depth >= max_depth or len(yy) < 2 * min_leaf or imp < 1e-6:
        return {"leaf": True, "p": float(yy.mean()) if len(yy) else 0.0}
    feat_idx = rng.choice(feat_pool, size=max(1, int(np.sqrt(len(feat_pool)))), replace=False)
    sp = best_split(Xmat, yy, feat_idx, criterion)
    if sp is None or sp[1] is None or sp[2].sum() < min_leaf or sp[3].sum() < min_leaf:
        return {"leaf": True, "p": float(yy.mean()) if len(yy) else 0.0}
    f, thr, left, right = sp
    return {"leaf": False, "f": f, "thr": float(thr),
            "left": build_tree(Xmat[left], yy[left], depth + 1, max_depth, min_leaf, feat_pool, criterion),
            "right": build_tree(Xmat[right], yy[right], depth + 1, max_depth, min_leaf, feat_pool, criterion)}

def tree_pred(tree, Xmat):
    out = np.empty(len(Xmat))
    for i in range(len(Xmat)):
        t = tree
        while not t["leaf"]:
            t = t["left"] if Xmat[i, t["f"]] <= t["thr"] else t["right"]
        out[i] = t["p"]
    return out

def rf_fit(Xtr, ytr, Xte, n_trees=120, max_depth=6, min_leaf=20):
    feat_pool = list(range(Xtr.shape[1]))
    trees = []
    for _ in range(n_trees):
        idx = rng.choice(len(ytr), len(ytr), replace=True)
        trees.append(build_tree(Xtr[idx], ytr[idx], 0, max_depth, min_leaf, feat_pool, "gini"))
    preds = np.mean([tree_pred(t, Xte) for t in trees], axis=0)
    return preds

# GBM: additive shallow trees on log-odds residuals, with 1-D line search per step
def gbm_fit(Xtr, ytr, Xte, n_trees=120, max_depth=3, min_leaf=20, lr=0.1):
    base = np.log((ytr.mean() + 1e-6) / (1 - ytr.mean() + 1e-6))
    f = np.full(len(Xtr), base)
    fte = np.full(len(Xte), base)
    feat_pool = list(range(Xtr.shape[1]))
    for _ in range(n_trees):
        p_hat = 1 / (1 + np.exp(-f))
        resid = ytr - p_hat
        tree = build_tree(Xtr, resid, 0, max_depth, min_leaf, feat_pool, "var")
        upd_tr = tree_pred(tree, Xtr)
        upd_te = tree_pred(tree, Xte)
        # line search: choose step that minimises logistic loss on train
        best_ls, best_loss = 0.0, float("inf")
        steps = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
        for s in steps:
            fs = f + s * upd_tr
            ps = 1 / (1 + np.exp(-fs))
            ps = np.clip(ps, 1e-12, 1 - 1e-12)
            loss = -np.mean(ytr * np.log(ps) + (1 - ytr) * np.log(1 - ps))
            if loss < best_loss:
                best_loss, best_ls = loss, s
        f += best_ls * upd_tr
        fte += best_ls * upd_te
    return 1 / (1 + np.exp(-fte))

# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------
def evaluate(name, fit_fn, X, y, n_folds=5, seed=42):
    r = np.random.default_rng(seed)
    folds = r.choice(n_folds, size=len(y))
    cv = []
    for kf in range(n_folds):
        te = folds == kf; tr = ~te
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
        prob = fit_fn((X[tr]-mu)/sd, y[tr], (X[te]-mu)/sd)
        prec, rec, f1, f2 = prf(y[te], prob)
        cv.append([prec, rec, f1, f2, roc_auc(y[te], prob)])
    cv = np.array(cv)
    # holdout 80/20
    idx = np.arange(len(y))
    r2 = np.random.default_rng(7)
    r2.shuffle(idx)
    cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
    prob = fit_fn((X[tr]-mu)/sd, y[tr], (X[te]-mu)/sd)
    prec, rec, f1, f2 = prf(y[te], prob)
    ho = [prec, rec, f1, f2, roc_auc(y[te], prob)]
    res = {"cv_mean": cv.mean(0).tolist(), "cv_std": cv.std(0).tolist(),
           "holdout": ho}
    print(f"{name:14s} CV  P={cv[:,0].mean():.3f} R={cv[:,1].mean():.3f} "
          f"F1={cv[:,2].mean():.3f} F2={cv[:,3].mean():.3f} AUC={cv[:,4].mean():.3f} | "
          f"HO P={ho[0]:.3f} R={ho[1]:.3f} F1={ho[2]:.3f} F2={ho[3]:.3f} AUC={ho[4]:.3f}")
    return res

print("Features:", FEATS, "| n =", n, "| distress rate =", round(y.mean(), 4))
results = {}
results["Logistic Regression (scipy)"] = evaluate("LogReg", logreg_fit, X, y)
results["Random Forest (numpy)"] = evaluate("RandomForest",
    lambda a, b, c: rf_fit(a, b, c), X, y)
results["GBM / XGBoost-like (numpy)"] = evaluate("GBM",
    lambda a, b, c: gbm_fit(a, b, c), X, y)

# Save predictions from best-by-F2 on holdout? We save CV-averaged metrics.
metrics = {"features": FEATS,
           "n": int(n), "distress_rate": float(y.mean()),
           "metrics": {k: {"cv_mean": v["cv_mean"], "cv_std": v["cv_std"],
                          "holdout": v["holdout"]} for k, v in results.items()},
           "metric_order": ["precision", "recall", "f1", "f2", "roc_auc"]}
with open(f"{OUT}/model_comparison.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\nSaved", f"{OUT}/model_comparison.json")
