def detect_drift(**ctx):
    import os

    import pandas as pd
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    REF_PATH = os.environ["REF_PATH"]
    REPORT_PATH = os.environ["REPORT_PATH"]

    cleaned_path = ctx["ti"].xcom_pull(
        key="cleaned_path", task_ids="load_and_clean_data"
    )

    current_df = pd.read_parquet(cleaned_path)

    if not os.path.exists(REF_PATH):
        print("No reference data — first run, skipping drift check.")
        ctx["ti"].xcom_push(key="drift_detected", value=True)
        ctx["ti"].xcom_push(key="drift_share", value=0.0)
        return

    ref_df = pd.read_parquet(REF_PATH)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df[["status_id"]], current_data=current_df[["status_id"]])

    report.save_html(REPORT_PATH)
    result = report.as_dict()

    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    share = result["metrics"][0]["result"]["share_of_drifted_columns"]
    print(f"Data drift detected: {drift_detected}  (share={share:.2%})")
    # Todo if drift not detected skip retrain

    ctx["ti"].xcom_push(key="drift_detected", value=drift_detected)
    ctx["ti"].xcom_push(key="drift_share", value=share)
    ctx["ti"].xcom_push(key="drift_report_path", value=REPORT_PATH)

def data_drift_check(**ctx):
    return ctx["ti"].xcom_pull(
        key="drift_detected", task_ids="detect_data_drift"
    )
