from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import evaluate
import torch

tokenizer = AutoTokenizer.from_pretrained("pborchert/BusinessBERT")

ogpath = "multichannel.csv"  
dataset = load_dataset('csv', data_files={'train': "train_" + ogpath, 'test': "test_" + ogpath})


def tokenize_function(examples):
    return tokenizer(examples["paragraph"], padding="max_length", truncation=True, max_length=512)


tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=16)


small_train_dataset = tokenized_datasets["train"]
small_eval_dataset = tokenized_datasets["test"]

# Load the model for regression (num_labels 1 for regression)
model = AutoModelForSequenceClassification.from_pretrained("pborchert/BusinessBERT", num_labels=3)

training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
)

metric = evaluate.load("mse")

# Function to compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.squeeze()  # For regression, squeeze to remove unnecessary dimensions
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
