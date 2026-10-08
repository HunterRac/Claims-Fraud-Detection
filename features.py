"""Load the vehicle claims data and turn it into model features.

Most fields in this dataset arrive pre-bucketed ("more than 30", "2 to 4").
Ordered buckets are mapped to a representative number so models can use their
order; unordered fields stay categorical.
"""
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "fraud_oracle.csv"
# Mirror of the Kaggle "Vehicle Insurance Claim Fraud Detection" dataset (fraud_oracle.csv).
DATA_URL = ("https://raw.githubusercontent.com/MohammadRouintan/Vehicle-Insurance-Claim-Fraud-Detection-Classification/"
            "HEAD/Datasets/fraud_oracle.csv")
TARGET = "FraudFound_P"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Bucket label -> representative value (midpoint, or a sensible cap for open-ended buckets).
ORDINAL_MAPS = {
    "VehiclePrice": {  # $000s
        "less than 20000": 15, "20000 to 29000": 25, "30000 to 39000": 35,
        "40000 to 59000": 50, "60000 to 69000": 65, "more than 69000": 75,
    },
    "Days_Policy_Accident": {"none": 0, "1 to 7": 4, "8 to 15": 11.5, "15 to 30": 22.5, "more than 30": 45},
    "Days_Policy_Claim": {"none": 0, "1 to 7": 4, "8 to 15": 11.5, "15 to 30": 22.5, "more than 30": 45},
    "PastNumberOfClaims": {"none": 0, "1": 1, "2 to 4": 3, "more than 4": 6},
    "AgeOfVehicle": {
        "new": 0, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5,
        "6 years": 6, "7 years": 7, "more than 7": 9,
    },
    "AgeOfPolicyHolder": {
        "16 to 17": 16.5, "18 to 20": 19, "21 to 25": 23, "26 to 30": 28, "31 to 35": 33,
        "36 to 40": 38, "41 to 50": 45.5, "51 to 65": 58, "over 65": 70,
    },
    "NumberOfSuppliments": {"none": 0, "1 to 2": 1.5, "3 to 5": 4, "more than 5": 7},
    # Years since the insured's address changed; "no change" treated as long ago.
    "AddressChange_Claim": {
        "under 6 months": 0.25, "1 year": 1, "2 to 3 years": 2.5, "4 to 8 years": 6, "no change": 10,
    },
    "NumberOfCars": {"1 vehicle": 1, "2 vehicles": 2, "3 to 4": 3.5, "5 to 8": 6.5, "more than 8": 9},
}

CATEGORICAL = [
    "Make", "AccidentArea", "Sex", "MaritalStatus", "Fault", "PolicyType", "VehicleCategory",
    "BasePolicy", "AgentType", "PoliceReportFiled", "WitnessPresent", "DayOfWeek",
    "DayOfWeekClaimed", "Month",
]
NUMERIC = list(ORDINAL_MAPS) + [
    "Age", "Deductible", "DriverRating", "WeekOfMonth", "WeekOfMonthClaimed",
    "report_lag_weeks",
]
FEATURES = CATEGORICAL + NUMERIC


def load_raw(path=DATA_PATH):
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urlretrieve(DATA_URL, path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    # One row has "0" for both claim-date fields; it can't be placed in time, so drop it.
    return df[df["MonthClaimed"] != "0"].reset_index(drop=True)


def report_lag_weeks(df):
    """Approximate weeks between the accident and the claim being reported.

    Only month and week-of-month are given, so this is accurate to about a week.
    A claim month earlier than the accident month means the claim rolled into the next year.
    """
    m_acc = df["Month"].map(MONTHS.index)
    m_clm = df["MonthClaimed"].map(MONTHS.index)
    months = (m_clm - m_acc) % 12
    weeks = months * 52 / 12 + (df["WeekOfMonthClaimed"] - df["WeekOfMonth"])
    return weeks.clip(lower=0).round(1)


def build_features(raw):
    X = pd.DataFrame(index=raw.index)
    for col in CATEGORICAL:
        X[col] = raw[col].astype("category")
    for col, mapping in ORDINAL_MAPS.items():
        X[col] = raw[col].map(mapping).astype(float)
    # Age 0 appears in about 320 rows, always alongside AgeOfPolicyHolder "16 to 17": it's missing, not a newborn.
    X["Age"] = raw["Age"].replace(0, np.nan).astype(float)
    for col in ["Deductible", "DriverRating", "WeekOfMonth", "WeekOfMonthClaimed"]:
        X[col] = raw[col].astype(float)
    X["report_lag_weeks"] = report_lag_weeks(raw)
    return X[FEATURES]


def red_flag_score(raw):
    """Rule-based scorecard, standing in for the red-flag checklist a claims team uses without a model.

    One point per classic indicator; higher means more suspicious.
    """
    flags = pd.DataFrame({
        "policyholder_at_fault": raw["Fault"] == "Policy Holder",
        "first_party_coverage": raw["BasePolicy"] != "Liability",
        "no_police_report": raw["PoliceReportFiled"] == "No",
        "no_witness": raw["WitnessPresent"] == "No",
        "recent_address_change": raw["AddressChange_Claim"].isin(["under 6 months", "1 year", "2 to 3 years"]),
        "new_policy": raw["Days_Policy_Claim"] != "more than 30",
        "young_driver": raw["AgeOfPolicyHolder"].isin(["16 to 17", "18 to 20", "21 to 25"]),
        "late_report": report_lag_weeks(raw) >= 4,
    })
    return flags.sum(axis=1)


def load(path=DATA_PATH):
    raw = load_raw(path)
    return raw, build_features(raw), raw[TARGET].astype(int)
