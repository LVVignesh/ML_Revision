# ==========================================================
# BERT FINETUNING WITH HUGGINGFACE
# IMDB SENTIMENT CLASSIFICATION
# ==========================================================

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, roc_auc_score

# ==========================================================
# STEP 1: LOAD DATASET
# ==========================================================

dataset = load_dataset("imdb")

# Use smaller subset for faster training
train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
test_dataset = dataset["test"].shuffle(seed=42).select(range(1000))

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))


# ==========================================================
# STEP 2: LOAD PRETRAINED TOKENIZER
# ==========================================================

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")


# ==========================================================
# STEP 3: LOAD PRETRAINED BERT MODEL
# ==========================================================

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)


# ==========================================================
# STEP 4: DEFINE METRICS
# ==========================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

    return {
        "accuracy": accuracy_score(labels, predictions),
        "roc_auc": roc_auc_score(labels, probs)
    }


# ==========================================================
# STEP 5: TRAINING ARGUMENTS
# ==========================================================

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    do_train=True,
    do_eval=True
)




# ==========================================================
# STEP 6: TRAINER
# ==========================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# ==========================================================
# STEP 7: FINETUNE
# ==========================================================

trainer.train()

results = trainer.evaluate()

print("\nFINAL RESULTS")
print(results)
