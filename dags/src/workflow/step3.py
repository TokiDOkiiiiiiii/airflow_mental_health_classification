def retrain_model(**ctx):
    import json
    import os
    import sys

    import mlflow
    import pandas as pd
    import torch
    import torch.optim as optim
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from src.utility.dataset import MentalHealthDataset
    from src.utility.wrapper import BertSentimentWrapper
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    MODEL_NAME = os.environ["MODEL_NAME"]
    MLFLOW_URI = os.environ["MLFLOW_TRACKING_URI"]
    REGISTERED_MODEL = os.environ["REGISTERED_MODEL_NAME"]
    EPOCHS = int(os.environ["EPOCHS"])
    BATCH_SIZE = int(os.environ["BATCH_SIZE"])
    LEARNING_RATE = float(os.environ["LEARNING_RATE"])
    MAX_LEN = int(os.environ["MAX_LEN"])
    WEIGHT_DECAY = float(os.environ["WEIGHT_DECAY"])
    EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
    OUTPUT_PATH = os.environ["OUTPUT_PATH"]
    AWS_KEY = os.environ["AWS_ACCESS_KEY_ID"]

    SEED = 42
    ID2LABEL = {0: "Normal", 1: "Anxiety", 2: "Depression", 3: "Suicidal"}

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    best_model_path = os.path.join(OUTPUT_PATH, "best_model")

    # MLflow: restore soft-deleted experiment if needed
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    use_pin = torch.cuda.is_available()
    print(f"Training device : {device}")
    print(f"GPU             : {gpu_name}")
    if torch.cuda.is_available():
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"VRAM            : {vram} GB")
        print(f"Model           : {MODEL_NAME}  batch={BATCH_SIZE}")

    # Data
    cleaned_path = ctx["ti"].xcom_pull(
        key="cleaned_path", task_ids="load_and_clean_data"
    )
    df = pd.read_parquet(cleaned_path)
    x, y = df["text"], df["status_id"]
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.1, stratify=y, random_state=SEED
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_loader = DataLoader(
        MentalHealthDataset(x_train, y_train, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=use_pin,
    )
    val_loader = DataLoader(
        MentalHealthDataset(x_val, y_val, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE,
        pin_memory=use_pin,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Warm start: load Production model if available
    print(f"Cold start: loading {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    drift_share = (
        ctx["ti"].xcom_pull(key="drift_share", task_ids="detect_data_drift") or 0.0
    )
    f1 = 0.0

    with mlflow.start_run(run_name=f"quarterly_retrain_{ctx['ds_nodash']}") as run:
        mlflow.log_params(
            {
                "model_base": MODEL_NAME,
                "n_epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "max_len": MAX_LEN,
                "train_rows": len(x_train),
                "val_rows": len(x_val),
                "drift_share": drift_share,
                "device": str(device),
                "gpu_name": gpu_name,
            }
        )

        best_val_loss = float("inf")

        for epoch in range(EPOCHS):
            print(f"\n── Epoch {epoch + 1}/{EPOCHS} starting ──")

            # Train
            model.train()
            train_loss = 0.0
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                out = model(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                out.loss.backward()
                optimizer.step()
                train_loss += out.loss.item()
                if (i + 1) % 100 == 0:
                    print(
                        f"  [train] batch {i + 1}/{len(train_loader)}  "
                        f"loss={out.loss.item():.4f}"
                    )

            # Validate
            model.eval()
            val_loss, all_preds, all_labels = 0.0, [], []
            with torch.no_grad():
                for batch in val_loader:
                    out = model(
                        batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device),
                    )
                    val_loss += out.loss.item()
                    all_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
                    all_labels.extend(batch["labels"].numpy())

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="macro")

            print(
                f"Epoch {epoch + 1}/{EPOCHS}  "
                f"train_loss={avg_train:.4f}  "
                f"val_loss={avg_val:.4f}  "
                f"acc={acc:.4f}  "
                f"macro_f1={f1:.4f}"
            )

            mlflow.log_metrics(
                {
                    "train_loss": avg_train,
                    "val_loss": avg_val,
                    "val_accuracy": acc,
                    "val_macro_f1": f1,
                },
                step=epoch,
            )

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                model.save_pretrained(best_model_path)
                with open(os.path.join(best_model_path, "id2label.json"), "w") as f:
                    json.dump(ID2LABEL, f)
                print("New model saved")

        model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
        mlflow.log_metric("best_val_loss", best_val_loss)

        drift_report = ctx["ti"].xcom_pull(
            key="drift_report_path", task_ids="detect_data_drift"
        )
        if drift_report and os.path.exists(drift_report):
            mlflow.log_artifact(drift_report, artifact_path="drift")

        # For deployment
        artifacts = {"model_dir": best_model_path}
        conda_env = {
            "channels": ["defaults", "conda-forge"],
            "dependencies": [
                "python=3.10",
                "pip",
                {
                    "pip": [
                        "mlflow>=3.10",
                        "torch>=2.10",
                        "transformers>=4.38",
                        "pandas",
                        "numpy",
                        "scikit-learn",
                    ]
                },
            ],
            "name": "mental_health_classification_env",
        }

        print("Uploading model to MLflow (S3 artifact store)...")

        model_info = mlflow.pyfunc.log_model(
            artifact_path="bert_sentiment_model",
            python_model=BertSentimentWrapper(),
            artifacts=artifacts,
            conda_env=conda_env,
            registered_model_name=REGISTERED_MODEL,
        )
        print("Model uploaded ✅")

        ctx["ti"].xcom_push(
            key="new_model_version", value=model_info.registered_model_version
        )
        ctx["ti"].xcom_push(key="new_run_id", value=run.info.run_id)
        print(f"Run complete → run_id={run.info.run_id}")
        torch.cuda.empty_cache()
