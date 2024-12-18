import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from datasets import load_dataset
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import torchmetrics
import wandb
import optuna

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate classification report for multi-class
def generate_classification_report(model, dataloader, num_classes):
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
Classification Report:
    Accuracy: {accuracy:.4f}
    Precision (Macro): {precision:.4f}
    Recall (Macro): {recall:.4f}
    F1 Score (Macro): {f1:.4f}
    """

    print(report)

    # Append the report to a text file
    with open("classification_report.txt", "a") as f:
        f.write(report + "\n")

# Define the model
class BertClassifier(nn.Module):
    def __init__(self, num_labels=1, freeze_bert=False):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
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
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        logits = self.cls_head(cls_output)
        return logits

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load dataset and preprocess
ogpath = "combined.csv"
dataset = load_dataset('csv', data_files={'train': "train_" + ogpath, 'test': "test_" + ogpath})

# Truncate dataset to the first entries
def truncate_dataset(dataset):
    return dataset.select(range(min(64, len(dataset))))

dataset = {k: truncate_dataset(v) for k, v in dataset.items()}

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["paragraph"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = {split: data.map(tokenize_function, batched=True) for split, data in dataset.items()}
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

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

train_data = CustomDataset(train_dataset)
test_data = CustomDataset(test_dataset)

# Oversampling to balance labels
train_labels = [item['label'] for item in train_dataset]
label_counts = Counter(train_labels)
print("Original label distribution:", label_counts)

sampler = RandomOverSampler()
train_indices = list(range(len(train_labels)))
resampled_indices, resampled_labels = sampler.fit_resample(torch.tensor(train_indices).view(-1, 1), torch.tensor(train_labels))
resampled_indices = resampled_indices.flatten().tolist()

resampled_train_data = torch.utils.data.Subset(train_data, resampled_indices)
resampled_label_counts = Counter(resampled_labels)
print("Resampled label distribution:", resampled_label_counts)

# DataLoader
batch_size = 16
train_dataloader = DataLoader(resampled_train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Training function
def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, patience=3):
    model.to(device)
    mse_loss_fn = nn.MSELoss()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            model.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = mse_loss_fn(logits.view(-1), labels.float())
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
                val_loss += mse_loss_fn(logits.view(-1), labels.float()).item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return best_loss

# Optuna hyperparameter optimization
def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-4)
    eps = trial.suggest_loguniform("eps", 1e-8, 1e-6)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    train_dataloader = DataLoader(resampled_train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = BertClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps)
    total_steps = len(train_dataloader) * 10  # Higher default epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    val_loss = train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, epochs=10)
    with open("classification_report.txt", "a") as f:
        f.write(f"Run Parameters:\n lr: {lr}, eps: {eps}, batch_size: {batch_size}\n\n")
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best hyperparameters:", study.best_params)

# Final evaluation
model = BertClassifier(num_labels=3).to(device)  # Update num_labels to match your dataset
optimizer = torch.optim.AdamW(model.parameters(), lr=study.best_params['lr'], eps=study.best_params['eps'])
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 10)
train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, epochs=10)
generate_classification_report(model, test_dataloader, num_classes=3)  # Update num_classes to match dataset
