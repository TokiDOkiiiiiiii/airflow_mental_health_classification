import mlflow
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

def score_model(model, test_df):
    LABEL2ID = {"Normal": 0, "Anxiety": 1, "Depression": 2, "Suicidal": 3}
    all_labels = (test_df["status"].map(LABEL2ID)).tolist()
    prob = model.predict(test_df)

    all_preds = np.argmax(prob, 1).tolist()
    return f1_score(all_labels, all_preds, average="macro")

load_dotenv()
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "")
TEST_CSV = "data/mental_health_combined_test.csv"
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")
mlflow.set_tracking_uri(MLFLOW_URI)
test_df = pd.read_csv(TEST_CSV)[:10]
print("load new model")
new_model = mlflow.pyfunc.load_model(
    f"models:/{REGISTERED_MODEL_NAME}/latest"
)
new_f1 = score_model(new_model, test_df)
print(new_f1)
