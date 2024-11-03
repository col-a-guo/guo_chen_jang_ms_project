from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import evaluate
import torch

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("pborchert/BusinessBERT")

ogpath = "multichannel.csv"  
# Load the dataset
dataset = load_dataset('csv', data_files={'train': "train_" + ogpath, 'test': "test_" + ogpath})

# Function to tokenize the text
def tokenize_function(examples):
    return tokenizer(examples["paragraph"], padding="max_length", truncation=True)

# Tokenize the datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Shuffle and select a small subset of the dataset
small_train_dataset = tokenized_datasets["train"]
small_eval_dataset = tokenized_datasets["test"]

# Load the model for regression (change num_labels to 1 for regression)
model = AutoModelForSequenceClassification.from_pretrained("pborchert/BusinessBERT", num_labels=1)

# Define training arguments
training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
)

# Load evaluation metric (mean squared error for regression)
metric = evaluate.load("mse")

# Function to compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.squeeze()  # For regression, squeeze to remove unnecessary dimensions
    return metric.compute(predictions=predictions, references=labels)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
