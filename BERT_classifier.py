import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import requests

print("Loading CSV data...")
df = pd.read_csv(
    "C:/Users/alino/Downloads/Telegram Desktop/reviews.csv",
    sep="\t"
)
print(df.head())

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}

df = df[df["sentiment"].isin(label2id.keys())]
df["label_id"] = df["sentiment"].map(label2id)
print("Количество NaN в label_id:", df["label_id"].isna().sum())

X_train, X_val, y_train, y_val = train_test_split(
    df["review"].values,
    df["label_id"].values,
    test_size=0.2,
    random_state=42,
    stratify=df["label_id"]
)

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize_progress(texts):
    encodings = []
    for t in tqdm(texts, desc="Tokenizing"):
        enc = tokenizer(t, padding='max_length', truncation=True, max_length=128)
        encodings.append(enc)
    input_ids = torch.tensor([e['input_ids'] for e in encodings])
    attention_mask = torch.tensor([e['attention_mask'] for e in encodings])
    return {"input_ids": input_ids, "attention_mask": attention_mask}

train_encodings = tokenize_progress(X_train)
val_encodings = tokenize_progress(X_val)

class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewsDataset(train_encodings, y_train)
val_dataset = ReviewsDataset(val_encodings, y_val)

def download_file(url, local_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(local_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(local_path)}",
        total=total, unit='iB', unit_scale=True
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

cache_dir = "./bert_cache"
os.makedirs(cache_dir, exist_ok=True)
model_name = "bert-base-multilingual-cased"
model_file = os.path.join(cache_dir, "pytorch_model.bin")

if not os.path.exists(model_file):
    print("Downloading BERT weights with progress bar...")
    url = "https://huggingface.co/bert-base-multilingual-cased/resolve/main/pytorch_model.bin"
    download_file(url, model_file)
else:
    print("BERT weights already downloaded.")

print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    cache_dir=cache_dir
)
print("Model loaded.")

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    eval_steps=100,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

trainer.train()

logs = trainer.state.log_history
train_loss, eval_loss = [], []

for log in logs:
    if "loss" in log and "epoch" in log:
        train_loss.append(log["loss"])
    if "eval_loss" in log:
        eval_loss.append(log["eval_loss"])

plt.figure(figsize=(8, 5))
plt.plot(train_loss, label="Train loss", marker="o")
plt.plot(eval_loss, label="Validation loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.savefig("loss_plot.png", dpi=300)
plt.show()

preds = trainer.predict(val_dataset)
y_pred = np.argmax(preds.predictions, axis=1)
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred, target_names=label2id.keys()))

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return id2label[pred]

print(predict_sentiment("Отличный сервис, обязательно вернусь!"))
