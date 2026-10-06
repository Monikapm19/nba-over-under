"""Train/evaluate Logistic Regression, Random Forest (baseline), XGBoost, Neural Net on OVER vs UNDER."""
import json, itertools, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
from features import build_dataset, split, FEATURES

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

X = build_dataset()
tr, va, te = split(X)
sizes = {"all_games_after_dropping_pushes": len(X), "train_2008_2022": len(tr),
         "val_2023": len(va), "test_2024_2026": len(te)}
for n, df in [("train", tr), ("val", va), ("test", te)]:
    sizes[f"{n}_over_rate"] = round(float(df.over.mean()), 4)
print(json.dumps(sizes, indent=1))
Xtr, ytr, Xva, yva, Xte, yte = tr[FEATURES], tr.over, va[FEATURES], va.over, te[FEATURES], te.over


def pipe(clf, scale=True):
    steps = [SimpleImputer(strategy="median")]  # imputer is fit on TRAIN only
    if scale:
        steps.append(StandardScaler())
    return make_pipeline(*steps, clf)


# small hyper-parameter grids, selected on VALIDATION ROC-AUC (test set never used for tuning)
GRIDS = {
    "Logistic Regression": (
        lambda p: pipe(LogisticRegression(C=p["C"], max_iter=2000)),
        [{"C": c} for c in (0.001, 0.01, 0.1, 1.0)]),
    "Random Forest (baseline)": (
        lambda p: pipe(RandomForestClassifier(n_estimators=300, max_depth=p["d"], min_samples_leaf=p["l"],
                                              n_jobs=-1, random_state=SEED), scale=False),
        [{"d": d, "l": l} for d in (4, 8, None) for l in (10, 50)]),
    "XGBoost": (
        lambda p: pipe(XGBClassifier(n_estimators=p["n"], max_depth=p["d"], learning_rate=0.03,
                                     subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                                     n_jobs=4, eval_metric="logloss"), scale=False),
        [{"n": n, "d": d} for n in (100, 300) for d in (2, 4)]),
    "Neural Network (MLP)": (
        lambda p: pipe(MLPClassifier(hidden_layer_sizes=p["h"], alpha=p["a"], max_iter=300,
                                     early_stopping=True, validation_fraction=0.1,
                                     n_iter_no_change=10, random_state=SEED)),
        [{"h": h, "a": a} for h in ((32,), (64, 32)) for a in (1e-2, 1.0)]),
}


def metrics(y, p, prob):
    return dict(accuracy=accuracy_score(y, p), precision=precision_score(y, p, zero_division=0),
                recall=recall_score(y, p), f1=f1_score(y, p),
                roc_auc=roc_auc_score(y, prob) if prob is not None else np.nan)


rows, cms, chosen, models = [], {}, {}, {}
for name, (mk, grid) in GRIDS.items():
    best = max(((roc_auc_score(yva, mk(p).fit(Xtr, ytr).predict_proba(Xva)[:, 1]), p) for p in grid),
               key=lambda z: z[0])
    val_auc, p = best
    m = mk(p).fit(Xtr, ytr)  # final model trained on the training seasons only
    chosen[name] = {"params": {k: str(v) for k, v in p.items()}, "val_auc": round(val_auc, 4)}
    models[name] = m
    for split_name, Xs, ys in (("validation", Xva, yva), ("test", Xte, yte)):
        prob = m.predict_proba(Xs)[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append(dict(model=name, split=split_name, **metrics(ys, pred, prob)))
        if split_name == "test":
            cms[name] = confusion_matrix(ys, pred)

# baselines (no learning): majority class of the TRAIN set, and a seeded coin flip
maj = int(ytr.mean() >= 0.5)
for split_name, ys in (("validation", yva), ("test", yte)):
    pm = np.full(len(ys), maj)
    rows.append(dict(model="Majority-class baseline", split=split_name, **metrics(ys, pm, None)))
    pc = np.random.RandomState(SEED).randint(0, 2, len(ys))
    rows.append(dict(model="Coin-flip baseline", split=split_name, **metrics(ys, pc, None)))
    if split_name == "test":
        cms["Majority-class baseline"] = confusion_matrix(ys, pm)
        cms["Coin-flip baseline"] = confusion_matrix(ys, pc)

R = pd.DataFrame(rows)
R.to_csv("results/metrics.csv", index=False)
test = R[R.split == "test"].set_index("model")
n = len(yte)
test["acc_95ci_halfwidth"] = 1.96 * np.sqrt(test.accuracy * (1 - test.accuracy) / n)
pd.set_option("display.width", 200)
print("\nVALIDATION")
print(R[R.split == "validation"].set_index("model").drop(columns="split").round(4))
print("\nTEST (n=%d)" % n)
print(test.drop(columns="split").round(4))
print("\nChosen hyper-parameters:", json.dumps(chosen, indent=1))
print("Break-even accuracy at standard -110 odds: 52.38%")
for k, v in cms.items():
    print(k, "confusion [[TN FP],[FN TP]] =", v.tolist())

# feature importance (Random Forest) and confusion-matrix figure
rf = models["Random Forest (baseline)"]
imp = pd.Series(rf[-1].feature_importances_, index=FEATURES).sort_values(ascending=False)
imp.to_csv("results/rf_feature_importance.csv")
print("\nTop RF features:\n", imp.head(8).round(4))
fig, ax = plt.subplots(1, 4, figsize=(14, 3.3))
for a, name in zip(ax, GRIDS):
    a.imshow(cms[name], cmap="Blues")
    for i, j in itertools.product(range(2), range(2)):
        a.text(j, i, cms[name][i, j], ha="center", va="center")
    a.set_xticks([0, 1], ["Under", "Over"])
    a.set_yticks([0, 1], ["Under", "Over"])
    a.set_title(name, fontsize=9)
    a.set_xlabel("Predicted")
    a.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("results/confusion_matrices.png", dpi=150)
json.dump({"sizes": sizes, "chosen": chosen, "features": FEATURES}, open("results/run_summary.json", "w"), indent=1)
test.to_csv("results/test_metrics_with_ci.csv")
