import mlflow


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
