import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from datasets import load_dataset
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import torchmetrics
import optuna
from sklearn.metrics import confusion_matrix
import random
import numpy as np

# Set the seed early
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # If using CUDA

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the versions
version_list = ["bert-uncased", "bert-uncased-pooling", "businessBERT"]

# Default hyperparameters for Optuna
default_lr = 5.841204543279205e-05
default_eps = 6.748313060587885e-08
default_batch_size = 32

# Function to generate classification report for multi-class
def generate_classification_report(model, dataloader, num_classes, epoch=None, version=None):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)  # Multi-class prediction
            all_preds.extend(preds.cpu())
            all_labels.extend(labels.cpu())

    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    # Update metrics to include task
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)(all_preds, all_labels)
    precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average='macro')(all_preds, all_labels)
    recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='macro')(all_preds, all_labels)
    f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')(all_preds, all_labels)

    report = f"""
Classification Report (Version: {version}, Epoch {epoch if epoch is not None else 'Final'}):
    Accuracy: {accuracy:.4f}
    Precision (Macro): {precision:.4f}
    Recall (Macro): {recall:.4f}
    F1 Score (Macro): {f1:.4f}
    """

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy(), labels=list(range(num_classes)))
    
    report += "\nConfusion Matrix:\n"
    report += "            Predicted\n"
    report += "           " + "    ".join(map(str, range(num_classes))) + "\n"
    report += "Actual\n"
    for i, row in enumerate(cm):
        report += f"      {i}   " + "    ".join(map(str, row)) + "\n"
    
    # Calculate and add TP, FP, TN, FN per class
    report += "\nPer-Class Metrics:\n"
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        report += f"  Class {i}: TP={tp}, FP={fp}, TN={tn}, FN={fn}\n"

    print(report)

    # Append the report to a text file
    with open("classification_report.txt", "a") as f:
        f.write(report + "\n")

    return f1 # Return F1 score

# Define the model architecture based on version
class BertClassifier(nn.Module):
    def __init__(self, version, num_labels=1, freeze_bert=False):
        super(BertClassifier, self).__init__()

        if version == "bert-uncased" or version == "bert-uncased-pooling":
            self.bert = AutoModel.from_pretrained('google-bert/bert-base-uncased')
        elif version == "businessBERT":
            self.bert = AutoModel.from_pretrained('pborchert/BusinessBERT')
        else:
           raise ValueError(f"Invalid model version: {version}")
        
        self.version = version  # Store the version
        
        if self.version == "bert-uncased-pooling":
            self.cls_head = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, num_labels)
            )
            self.pooling = nn.AdaptiveAvgPool1d(1) # Global average pooling layer

        else:
            self.cls_head = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, num_labels)
            )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

            if self.version == "bert-uncased-pooling":
                # Global average pooling
                last_hidden_state = outputs.last_hidden_state
                pooled_output = self.pooling(last_hidden_state.permute(0, 2, 1)).squeeze(-1)
                logits = self.cls_head(pooled_output)
            else:
                cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
                logits = self.cls_head(cls_output)
            return logits

# Function to load the correct tokenizer
def load_tokenizer(version):
    if version == "bert-uncased" or version == "bert-uncased-pooling":
        return AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    elif version == "businessBERT":
        return AutoTokenizer.from_pretrained('pborchert/BusinessBERT')
    else:
        raise ValueError(f"Invalid model version: {version}")

# Load dataset and preprocess
ogpath = "combined.csv"
dataset = load_dataset('csv', data_files={'train': "train_" + ogpath, 'test': "test_" + ogpath})

# Truncate dataset to the first entries
def truncate_dataset(dataset):
    k = round(len(dataset)*0.97)
    random_indices = random.sample(range(len(dataset)), k)
    return dataset.select(random_indices)

dataset = {k: truncate_dataset(v) for k, v in dataset.items()}

# Tokenize dataset
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["paragraph"], padding="max_length", truncation=True, max_length=512)

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = torch.tensor(item['input_ids'])
        attention_mask = torch.tensor(item['attention_mask'])
        label = torch.tensor(item['label'], dtype=torch.long)
        return input_ids, attention_mask, label

# Training function
def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, loss_fn, patience=3, num_classes=3, version=None):
    model.to(device)
    best_f1 = 0.0  # Initialize best f1 score
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            model.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels) #  Calculate loss using weighted CrossEntropyLoss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask, labels = [t.to(device) for t in batch]
                logits = model(input_ids, attention_mask)
                val_loss += loss_fn(logits, labels).item() # Calculate validation loss using weighted CrossEntropyLoss

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        
        # Generate and save the classification report every epoch
        f1_score = generate_classification_report(model, val_dataloader, num_classes, epoch=epoch+1, version=version)

        # Early stopping based on F1 score
        if f1_score > best_f1:
            best_f1 = f1_score
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return best_f1 # return the best f1 score

# Optuna hyperparameter optimization
def objective(trial, version, train_data, test_data, loss_fn):
    lr = trial.suggest_loguniform("lr", default_lr, default_lr) # Use defaults
    eps = trial.suggest_loguniform("eps", default_eps, default_eps) # Use defaults
    batch_size = trial.suggest_categorical("batch_size", [16, 32]) # Use defaults


    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = BertClassifier(version, num_labels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps)
    total_steps = len(train_dataloader) * 20
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    val_f1 = train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, epochs=10, loss_fn=loss_fn, version=version)
    with open("classification_report.txt", "a") as f:
        f.write(f"Run Parameters for {version}:\n lr: {lr}, eps: {eps}, batch_size: {batch_size}\n\n")
    return -val_f1 # Optuna minimizes, we want to maximize F1 so return negative F1


# Main loop
for version in version_list:
    print(f"\n----- Running with {version} -----")

    # Load the correct tokenizer for the current version
    tokenizer = load_tokenizer(version)
    tokenized_datasets = {split: data.map(lambda examples: tokenize_function(examples, tokenizer), batched=True) for split, data in dataset.items()}
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    train_data = CustomDataset(train_dataset)
    test_data = CustomDataset(test_dataset)

    # Oversampling to balance labels
    train_labels = [item['label'] for item in train_dataset]
    label_counts = Counter(train_labels)
    print("Original label distribution:", label_counts)
    
    # Calculate Class Weights
    class_weights = torch.tensor([1/2049, 1/188, 1/77], dtype=torch.float)
    min_weight = min(class_weights)
    max_weight = max(class_weights)
    geometric_mean_mult = (max_weight/min_weight)**0.5*3

    normalized_weights = class_weights*geometric_mean_mult/max_weight
    
    print(normalized_weights)
    loss_fn = nn.CrossEntropyLoss(weight=normalized_weights.to(device)) # Weighted Loss

    sampler = RandomOverSampler()
    train_indices = list(range(len(train_labels)))
    resampled_indices, resampled_labels = sampler.fit_resample(torch.tensor(train_indices).view(-1, 1), torch.tensor(train_labels))
    resampled_indices = resampled_indices.flatten().tolist()

    resampled_train_data = torch.utils.data.Subset(train_data, resampled_indices)
    resampled_label_counts = Counter(resampled_labels)
    print("Resampled label distribution:", resampled_label_counts)

    # Optuna Hyperparameter Tuning with reduced trials
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, version, resampled_train_data, test_data, loss_fn), n_trials=2) # Reduced Trials
    print("Best hyperparameters:", study.best_params)

    # Final evaluation
    model = BertClassifier(version, num_labels=3).to(device)  # Update num_labels to match dataset
    train_dataloader = DataLoader(resampled_train_data, batch_size=study.best_params['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=study.best_params['batch_size'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=study.best_params['lr'], eps=study.best_params['eps'])
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 10)
    train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, epochs=30, loss_fn=loss_fn, num_classes=3, version=version)
    generate_classification_report(model, test_dataloader, num_classes=3, version=version)