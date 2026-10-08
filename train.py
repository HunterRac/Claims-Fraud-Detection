"""Train and evaluate claims fraud models with a time-based split.

Train on 1994-95 claims, test on 1996 claims, the way a model built today gets used on
next year's claims. Hyperparameters are picked by cross-validation inside 1994-95, so the 1996 test set is
touched once.

Outputs (in outputs/): metrics.json, scored_claims.parquet (for the dashboard), and plots.
"""
import itertools
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL, NUMERIC, load, red_flag_score

# lgb.cv passes groups through to the pre-built folds; sklearn warns that it ignores them.
warnings.filterwarnings("ignore", message="The groups parameter is ignored")

SEED = 42
REVIEW_RATES = [0.05, 0.10, 0.20]  # share of claims the investigations unit can review
OUT = Path(__file__).parent / "outputs"


def logistic(C):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CATEGORICAL),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUMERIC),
    ])
    return make_pipeline(pre, LogisticRegression(C=C, max_iter=2000))


LGB_BASE = dict(objective="binary", learning_rate=0.02, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.5, random_state=SEED, verbose=-1)


def tune(X, y, groups):
    """Pick hyperparameters by 5-fold CV on the training years, scored by average precision.

    Folds are grouped by accident month. Claims from the same month share patterns, and a
    random split leaks them: the same model scores 0.26 AP with random folds but 0.16 with
    month-grouped folds, and the grouped number is the one that matches the 1996 test.
    """
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups))
    lr_cv = {C: cross_val_score(logistic(C), X, y, cv=folds, scoring="average_precision").mean()
             for C in [0.01, 0.03, 0.1, 0.3, 1.0]}
    best_lr = max(lr_cv, key=lr_cv.get)

    results = []
    for leaves, mcs, reg in itertools.product([7, 15, 31], [20, 50, 100], [1.0, 10.0]):
        params = dict(num_leaves=leaves, min_child_samples=mcs, reg_lambda=reg)
        cv = lgb.cv({**LGB_BASE, **params, "metric": "average_precision"}, lgb.Dataset(X, y),
                    num_boost_round=3000, folds=folds, callbacks=[lgb.early_stopping(200, verbose=False)])
        curve = cv["valid average_precision-mean"]
        best = int(np.argmax(curve))
        results.append((curve[best], {**params, "n_estimators": best + 1}))
    best_ap, best_gbm = max(results, key=lambda r: r[0])
    print(f"Tuned: logistic C={best_lr} (CV AP {lr_cv[best_lr]:.3f}); LightGBM {best_gbm} (CV AP {best_ap:.3f})")
    return best_lr, best_gbm, {"Logistic regression": lr_cv[best_lr], "LightGBM": best_ap}


def at_review_rate(y, score, rate, rng):
    """Precision and recall if investigators review the top `rate` share of claims by score."""
    k = int(round(rate * len(y)))
    order = np.lexsort((rng.random(len(y)), -score))  # random tie-break, matters for the integer rule score
    caught = y[order[:k]].sum()
    return caught / k, caught / y.sum()


def bootstrap_ap(y, score, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    aps = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum():
            aps.append(average_precision_score(y[idx], score[idx]))
    return np.percentile(aps, [2.5, 97.5])


def evaluate(y, scores):
    rng = np.random.default_rng(SEED)
    rows = {}
    for name, s in scores.items():
        lo, hi = bootstrap_ap(y, s)
        row = {"roc_auc": roc_auc_score(y, s), "avg_precision": average_precision_score(y, s),
               "avg_precision_ci": [lo, hi]}
        for rate in REVIEW_RATES:
            p, r = at_review_rate(y, s, rate, rng)
            row[f"precision@{rate:.0%}"], row[f"recall@{rate:.0%}"] = p, r
        rows[name] = row
    return rows


def display_value(raw_row, feature, value):
    """Human-readable value for a reason code: the original label where one exists."""
    if feature in raw_row.index:
        return str(raw_row[feature])
    return f"{value:.0f} wk"  # report_lag_weeks, the only derived feature


def reason_codes(model, X, raw, top=3):
    """Per-claim TreeSHAP contributions (log-odds) from LightGBM, plus top reasons pushing toward fraud."""
    contrib = model.booster_.predict(X, pred_contrib=True)
    contrib = pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index)
    reasons = []
    for i in X.index:
        c = contrib.loc[i]
        up = c[c > 0].nlargest(top)
        reasons.append("; ".join(f"{f} = {display_value(raw.loc[i], f, X.at[i, f])}" for f in up.index))
    return contrib, reasons


def plot_curves(y, scores, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    base = y.mean()
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        axes[0].plot(r, p, label=f"{name} (AP {average_precision_score(y, s):.3f})")
        order = np.argsort(-s, kind="stable")
        captured = np.cumsum(y[order]) / y.sum()
        axes[1].plot(np.arange(1, len(y) + 1) / len(y), captured, label=name)
    axes[0].axhline(base, color="grey", ls=":", label=f"random ({base:.3f})")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall, 1996 test claims")
    axes[1].plot([0, 1], [0, 1], color="grey", ls=":", label="random")
    for rate in REVIEW_RATES:
        axes[1].axvline(rate, color="lightgrey", lw=0.8)
    axes[1].set(xlabel="Share of claims reviewed", ylabel="Share of fraud caught",
                title="Fraud capture by review capacity", xlim=(0, 0.5))
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importance(contrib, path, top=15):
    imp = contrib.abs().mean().sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(imp.index, imp.values, color="#3b6ea5")
    ax.set(xlabel="Mean |SHAP contribution| (log-odds)", title="What drives the fraud score (1996 test claims)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    raw, X, y = load()
    years = raw["Year"].to_numpy()
    y = y.to_numpy()
    tr, te = years <= 1995, years == 1996
    print(f"Train 1994-95: {tr.sum():,} claims ({y[tr].mean():.1%} fraud); "
          f"test 1996: {te.sum():,} claims ({y[te].mean():.1%} fraud)")

    month = (raw["Year"].astype(str) + raw["Month"]).to_numpy()
    C, gbm_params, cv_ap = tune(X[tr], y[tr], month[tr])
    lr = logistic(C).fit(X[tr], y[tr])
    gbm = lgb.LGBMClassifier(**gbm_params, **LGB_BASE).fit(X[tr], y[tr])

    X_te, raw_te, y_te = X[te], raw[te], y[te]
    scores = {
        "Red-flag rules": red_flag_score(raw_te).to_numpy().astype(float),
        "Logistic regression": lr.predict_proba(X_te)[:, 1],
        "LightGBM": gbm.predict_proba(X_te)[:, 1],
    }
    metrics = evaluate(y_te, scores)
    table = pd.DataFrame(metrics).T.drop(columns="avg_precision_ci")
    print("\n1996 test results\n" + table.round(3).to_string())

    contrib, reasons = reason_codes(gbm, X_te, raw_te)
    plot_curves(y_te, scores, f"{OUT}/curves.png")
    plot_importance(contrib, f"{OUT}/importance.png")

    scored = raw_te.copy()
    scored["fraud_score"] = scores["LightGBM"]
    scored["red_flags"] = scores["Red-flag rules"].astype(int)
    scored["top_reasons"] = reasons
    scored["report_lag_weeks"] = X_te["report_lag_weeks"]
    scored = scored.join(contrib.add_prefix("shap__"))
    scored = scored.sort_values("fraud_score", ascending=False)
    scored.insert(0, "queue_rank", np.arange(1, len(scored) + 1))
    scored.to_parquet(f"{OUT}/scored_claims.parquet", index=False)

    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump({"train_years": [1994, 1995], "test_year": 1996,
                   "logistic_C": C, "lightgbm_params": gbm_params, "cv_avg_precision_1994_95": cv_ap,
                   "test_claims": int(te.sum()), "test_fraud": int(y_te.sum()),
                   "models": metrics}, f, indent=2, default=float)
    print(f"\nWrote {OUT}/metrics.json, scored_claims.parquet, curves.png, importance.png")


if __name__ == "__main__":
    main()
