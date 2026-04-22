import numpy as np
from sklearn.metrics import f1_score


def score_model(model, test_df):
    LABEL2ID = {"Normal": 0, "Negative": 1, "Very Negative": 2, "Suicidal": 3}
    all_labels = (test_df["status"].map(LABEL2ID)).tolist()
    prob = model.predict(test_df)
    all_preds = np.argmax(prob, 1).tolist()
    return f1_score(all_labels, all_preds, average="macro")
