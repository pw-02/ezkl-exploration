import os
from typing import Dict

import pandas as pd


def read_csv_first_row(file_path: str) -> Dict:
    if not os.path.exists(file_path):
        return {}

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return {}
        return df.iloc[0].to_dict()
    except Exception:
        return {}


def summarize_fft_report(file_path: str) -> Dict:
    if not os.path.exists(file_path):
        return {}

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return {}

        return {
            "fft_count": int(len(df)),
            "fft_largest": int(df["size"].max()),
            "fft_total_time(s)": float(df["duration(s)"].sum()),
            "fft_avg_time(s)": float(df["duration(s)"].mean()),
            "fft_device": str(df["device"].iloc[0]),
        }
    except Exception:
        return {}


def summarize_msm_report(file_path: str) -> Dict:
    if not os.path.exists(file_path):
        return {}

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return {}

        return {
            "msm_count": int(len(df)),
            "msm_largest": int(df["num_coeffs"].max()),
            "msm_total_time(s)": float(df["duration(s)"].sum()),
            "msm_avg_time(s)": float(df["duration(s)"].mean()),
            "msm_device": str(df["device"].iloc[0]),
        }
    except Exception:
        return {}