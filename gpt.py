from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import os

MODEL_NAME = r"C:\Users\alino\PycharmProjects\python1\my_model_local"

# проверяем наличие модели
if not os.path.isdir(MODEL_NAME):
    raise ValueError("Папка с моделью не найдена")

# токенизатор
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# загружаем данные
dataset = load_dataset(
    "text",
    data_files={
        "train": "train.txt",
        "validation": "valid.txt"
    }
)

# фильтруем пустые строки
dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

# функция токенизации
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=64)

# токенизация
tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

# модель
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# аргументы обучения
training_args = TrainingArguments(
    output_dir="my_model",          # модель
    logging_dir="training_logs",    # логи
    overwrite_output_dir=False,

    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,

    save_strategy="epoch",
    logging_steps=50,
    report_to="none"
)


# DataCollator
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=collator
)

# обучение
trainer.train()

# сохраняем модель и токенизатор
trainer.save_model("my_model")
tokenizer.save_pretrained("my_model")
