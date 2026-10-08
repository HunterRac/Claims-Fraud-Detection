"""Claims triage dashboard: a ranked investigation queue with per-claim explanations.

Shows the 1996 test claims as if they were this year's inbox, scored by the model trained on
1994-95. Run `python train.py` first, then `streamlit run app.py`.
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

OUT = Path(__file__).parent / "outputs"
SHOW_COLS = ["Make", "PolicyType", "VehiclePrice", "Fault", "AccidentArea", "AgeOfPolicyHolder",
             "Days_Policy_Claim", "PastNumberOfClaims", "AddressChange_Claim", "PoliceReportFiled",
             "WitnessPresent", "AgentType", "Deductible", "report_lag_weeks"]

st.set_page_config(page_title="Claims Fraud Triage", layout="wide")


@st.cache_data
def load():
    claims = pd.read_parquet(f"{OUT}/scored_claims.parquet")
    with open(f"{OUT}/metrics.json") as f:
        metrics = json.load(f)
    return claims, metrics


def capture(scores, y, k, seed=42):
    """Fraud caught in the top k claims by score (random tie-break, as in train.py)."""
    order = np.lexsort((np.random.default_rng(seed).random(len(y)), -scores))
    return int(y[order[:k]].sum())


claims, metrics = load()
base_rate = claims["FraudFound_P"].mean()
y = claims["FraudFound_P"].to_numpy()
total_fraud = int(y.sum())

with st.sidebar:
    st.header("Investigations unit")
    rate = st.slider("Review capacity (% of claims)", 1, 40, 10) / 100
    st.subheader("Economics (assumptions)")
    cost = st.number_input("Cost per investigation ($)", 0, 10_000, 400, step=50)
    saved = st.number_input("Avg. loss avoided per fraud caught ($)", 0, 100_000, 6_000, step=500)
    st.caption("The dataset has no claim amounts, so savings use these inputs. Adjust to your book.")
    show_outcome = st.toggle("Show known outcome (backtest)", value=True)

k = int(round(rate * len(claims)))
caught_model = capture(claims["fraud_score"].to_numpy(), y, k)
caught_rules = capture(claims["red_flags"].to_numpy().astype(float), y, k)
net_model = caught_model * saved - k * cost
net_rules = caught_rules * saved - k * cost

st.title("Claims Fraud Triage")
st.caption(f"{len(claims):,} auto claims from 1996, scored by a LightGBM model trained only on 1994–95 claims. "
           f"{total_fraud} ({base_rate:.1%}) were confirmed fraud.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Claims to review", f"{k:,}")
c2.metric("Fraud caught (model)", f"{caught_model} of {total_fraud}", f"{caught_model - caught_rules:+d} vs. red-flag rules")
c3.metric("Hit rate", f"{caught_model / k:.1%}", f"{caught_model / k / base_rate:.1f}x random", delta_color="off")
c4.metric("Net savings (model)", f"${net_model:,.0f}", f"${net_model - net_rules:+,.0f} vs. rules")

queue_tab, detail_tab, perf_tab = st.tabs(["Investigation queue", "Claim detail", "Model performance"])

queue = claims.head(k).copy()
queue["risk"] = queue["fraud_score"] / base_rate

with queue_tab:
    cols = ["queue_rank", "PolicyNumber", "risk", "red_flags", "PolicyType", "Fault", "top_reasons"]
    if show_outcome:
        cols.append("FraudFound_P")
    st.dataframe(
        queue[cols], hide_index=True, width="stretch", height=560,
        column_config={
            "queue_rank": st.column_config.NumberColumn("Rank", width="small"),
            "PolicyNumber": st.column_config.NumberColumn("Policy #", format="%d", width="small"),
            "risk": st.column_config.ProgressColumn("Risk vs. avg.", format="%.1fx", min_value=0,
                                                   max_value=float(claims["fraud_score"].max() / base_rate)),
            "red_flags": st.column_config.NumberColumn("Red flags", help="Rule-based checklist score, 0–8"),
            "top_reasons": st.column_config.TextColumn("Top reasons (model)", width="large"),
            "FraudFound_P": st.column_config.CheckboxColumn("Fraud?"),
        },
    )

with detail_tab:
    policy = st.selectbox("Claim in queue (rank: policy #)", queue["PolicyNumber"],
                          format_func=lambda p: f"#{int(queue.loc[queue.PolicyNumber == p, 'queue_rank'].iloc[0])}: policy {p}")
    claim = claims.loc[claims["PolicyNumber"] == policy].iloc[0]
    left, right = st.columns([1, 2])
    with left:
        st.metric("Risk vs. average claim", f"{claim['fraud_score'] / base_rate:.1f}x",
                  f"model score {claim['fraud_score']:.1%}", delta_color="off")
        if show_outcome:
            st.write("**Outcome:**", "Confirmed fraud" if claim["FraudFound_P"] else "Not fraud")
        st.dataframe(claim[SHOW_COLS].astype(str).rename("value"), width="stretch")
    with right:
        shap = claim.filter(like="shap__").astype(float)
        shap.index = shap.index.str.removeprefix("shap__")
        top = shap.reindex(shap.abs().sort_values(ascending=False).index).head(12)
        chart_df = pd.DataFrame({
            "feature": [f"{f} = {claim[f] if f in claim.index else ''}" for f in top.index],
            "contribution": top.values,
        })
        chart_df["direction"] = np.where(chart_df["contribution"] > 0, "raises risk", "lowers risk")
        st.subheader("Why this claim was flagged")
        st.altair_chart(
            alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("contribution:Q", title="SHAP contribution (log-odds)"),
                y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
                color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                               range=["#c0392b", "#2e86c1"]), title=None),
                tooltip=["feature", alt.Tooltip("contribution:Q", format="+.3f")],
            ).properties(height=380),
            width="stretch",
        )

with perf_tab:
    rows = {name: {"ROC AUC": m["roc_auc"], "Avg. precision": m["avg_precision"],
                   "AP 95% CI": f"{m['avg_precision_ci'][0]:.3f}–{m['avg_precision_ci'][1]:.3f}",
                   "Recall @ 10% reviewed": m["recall@10%"], "Precision @ 10% reviewed": m["precision@10%"]}
            for name, m in metrics["models"].items()}
    st.dataframe(pd.DataFrame(rows).T, width="stretch")
    st.image(f"{OUT}/curves.png")
    st.image(f"{OUT}/importance.png", width=620)
