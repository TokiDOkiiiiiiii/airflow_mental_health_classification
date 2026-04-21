def load_and_clean_data(**ctx):
    import os

    import pandas as pd
    import sys
    sys.path.insert(0, "/opt/airflow/dags/src/utility")
    from preprocessing import cleaning_pipeline

    TRAIN_PATH = os.environ["TRAIN_PATH"]
    DATA_PATH = os.environ["DATA_PATH"]

    df = pd.read_csv(TRAIN_PATH)
    cleaned = cleaning_pipeline(df)
    cleaned.to_parquet(DATA_PATH, index=False)
    ctx["ti"].xcom_push(key="cleaned_path", value=DATA_PATH)
    ctx["ti"].xcom_push(key="n_rows", value=len(cleaned))
    print(f"Cleaned dataset saved → {DATA_PATH}  ({len(cleaned)} rows)")
