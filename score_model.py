#!/usr/bin/env python3
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
         "Total financial expenses", "Operating cash flow", "Current taxes",
         "Alert Index"]  # derived/redundant vars dropped for coefficient stability

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
