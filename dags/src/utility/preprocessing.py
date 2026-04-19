# Data cleaning pipeline extracted directly from the BERT notebook.

import re
import string

import pandas as pd

STATUS_MAP = {"Normal": 0, "Negative": 1, "Very Negative": 2, "Suicidal": 3}
SEED = 42


def clean_text(text: str) -> str:
    """Exact replica of the notebook's clean_text function."""
    text = str(text)
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = text.replace("'", "")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return text


def cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Exact replica of the notebook's cleaning_pipeline function."""
    df = df.copy()
    if "Unique_ID" in df.columns:
        df = df.drop("Unique_ID", axis=1)
    df["status"] = df["status"].astype(str).str.strip()
    df["status_id"] = df["status"].map(STATUS_MAP)
    df = df.dropna(subset=["status_id"])
    df["status_id"] = df["status_id"].astype(int)
    df = df.drop("status", axis=1)
    df["text"] = df["text"].apply(clean_text)
    return df
