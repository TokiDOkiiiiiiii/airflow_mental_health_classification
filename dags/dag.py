"""
Runs every 3 months. Linear pipeline:
  1. Load & clean data
  2. Detect data drift
  3. Retrain DistilBERT
  4. Evaluate & promote
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from dotenv import load_dotenv

load_dotenv()

default_args = {
    "owner": "Guess Engineering",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="distilbert_mental_health_quarterly_retrain",
    description="Quarterly retrain of DistilBERT mental-health classifier",
    schedule="0 0 1 */3 *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["nlp", "distilbert", "mental-health", "mlops"],
) as dag:
    # Load & clean data
    from src.workflow.step1 import load_and_clean_data

    load_task = PythonOperator(
        task_id="load_and_clean_data",
        python_callable=load_and_clean_data,
    )

    # Detect data drift
    from src.workflow.step2 import detect_drift, data_drift_check

    drift_task = PythonOperator(
        task_id="detect_data_drift",
        python_callable=detect_drift,
    )
    drift_check = ShortCircuitOperator(
        task_id="data_drift_check",
        python_callable=data_drift_check
    )

    # Retrain DistilBERT
    def retrain_model(**ctx):
        import json
        import os
        import sys
        import mlflow
        import pandas as pd
        import torch
        import torch.optim as optim
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        sys.path.insert(0, "/opt/airflow/dags/src/utility")
        from dataset import MentalHealthDataset


        class BertSentimentWrapper(mlflow.pyfunc.PythonModel):
            """
            Wraps BERT + tokenizer so that `mlflow models serve` can use it.
            Input:  pandas DataFrame with a column named 'text'
            Output: pandas DataFrame with columns 'label_id' and 'label'
            """

            def load_context(self, context):
                import json
                import os

                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                model_dir = context.artifacts["model_dir"]
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                self.model.to(self.device)
                self.model.eval()

                with open(os.path.join(model_dir, "id2label.json")) as f:
                    self.id2label = {int(k): v for k, v in json.load(f).items()}

            def clean_text(self, text) -> str:
                import re
                import string

                text = str(text)
                text = text.lower()
                text = re.sub(r"<.*?>", "", text)
                text = re.sub(r"https?://\S+|www\.\S+", "", text)
                text = text.replace("'", "")
                text = text.translate(str.maketrans("", "", string.punctuation))
                text = re.sub(r"\d+", "", text)
                return text

            def log_prediction(self, text, prediction):
                print(f"{text}: {prediction}")

            def predict(self, context, model_input):
                import torch

                original_texts = model_input["text"].tolist()
                texts = list(map(self.clean_text, original_texts))
                encodings = self.tokenizer(
                    texts,
                    max_length=128,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)

                with torch.no_grad():
                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).logits

                prob = torch.softmax(logits, dim=1).cpu().numpy().tolist()

                for i in range(len(original_texts)):
                    self.log_prediction(original_texts[i], prob[i])
                return prob



        MODEL_NAME = os.environ["MODEL_NAME"]
        MLFLOW_URI = os.environ["MLFLOW_TRACKING_URI"]
        REGISTERED_MODEL = os.environ["REGISTERED_MODEL_NAME"]
        EPOCHS = os.environ["EPOCHS"]
        BATCH_SIZE = int(os.environ["BATCH_SIZE"])
        LEARNING_RATE = float(os.environ["LEARNING_RATE"])
        MAX_LEN = int(os.environ["MAX_LEN"])
        WEIGHT_DECAY = float(os.environ["WEIGHT_DECAY"])
        EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
        OUTPUT_PATH = os.environ["OUTPUT_PATH"]

        SEED = 42
        ID2LABEL = {0: "Normal", 1: "Negative", 2: "Very Negative", 3: "Suicidal"}
        LABEL2ID = {"Normal" : 0,"Negative" : 1,"Very Negative" : 2,"Suicidal": 3}

        os.makedirs(OUTPUT_PATH, exist_ok=True)
        best_model_path = os.path.join(OUTPUT_PATH, "best_model")

        # MLflow: restore soft-deleted experiment if needed
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data
        cleaned_path = ctx["ti"].xcom_pull(
            key="cleaned_path", task_ids="load_and_clean_data"
        )
        df = pd.read_parquet(cleaned_path)
        x, y = df["text"].tolist(), df["status_id"].tolist()
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.1, stratify=y, random_state=SEED
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        train_loader = DataLoader(
            MentalHealthDataset(x_train, y_train, tokenizer, MAX_LEN),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        val_loader = DataLoader(
            MentalHealthDataset(x_val, y_val, tokenizer, MAX_LEN),
            batch_size=BATCH_SIZE,
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
                }
            )

            best_val_loss = float("inf")
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            with open(os.path.join(best_model_path, "id2label.json"), "w") as f:
                json.dump(ID2LABEL, f)

            # Problem with training but i cant identify
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
                report = classification_report(
                    all_labels, all_preds,
                    target_names=list(LABEL2ID.keys()),
                    output_dict=True,
                )
                for label_name in LABEL2ID:
                    if label_name in report:
                        mlflow.log_metrics({
                            f"{label_name}_precision": report[label_name]["precision"],
                            f"{label_name}_recall": report[label_name]["recall"],
                            f"{label_name}_f1": report[label_name]["f1-score"],
                        })

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
                    tokenizer.save_pretrained(best_model_path)
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
                            "torch==2.10",
                            "transformers>=4.38",
                            "pandas",
                            "numpy==2.0.2",
                            "scikit-learn==1.6.1",
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
                infer_code_paths=True,
                conda_env=conda_env,
                registered_model_name=REGISTERED_MODEL,
            )
            print("Model uploaded ✅")

            ctx["ti"].xcom_push(
                key="new_model_version", value=model_info.registered_model_version
            )
            ctx["ti"].xcom_push(key="new_run_id", value=run.info.run_id)
            print(f"Run complete → run_id={run.info.run_id}")


    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
        execution_timeout=timedelta(hours=4),
    )

    # Evaluate & promote
    from src.workflow.step4 import evaluate_and_promote, new_production_check

    evaluate_task = PythonOperator(
        task_id="evaluate_and_promote",
        python_callable=evaluate_and_promote,
        execution_timeout=timedelta(hours=2),
    )
    production_check = ShortCircuitOperator(
        task_id="new_production_check",
        python_callable=new_production_check
    )


    from src.workflow.step5 import update_deployment

    update_deployment = PythonOperator(
        task_id="update_deployment",
        python_callable=update_deployment,
        execution_timeout=timedelta(hours=2),
    )

    load_task >> drift_task >> drift_check >> retrain_task >> evaluate_task >> production_check >> update_deployment
