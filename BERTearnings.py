import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_scheduler
from datasets import load_dataset
import wandb
import time
import os

# Initialize WandB
wandb.init(project="bert-hyperparameter-search", entity="collinguo-sacramento-state")

os.environ['WANDB_TIMEOUT'] = '60' # set timeout to 60 seconds
# Default mode
default_mode = 'multiclass'

class BertClassifier(nn.Module):
    def __init__(self, mode=default_mode, freeze_bert=False):
        super(BertClassifier, self).__init__()
        D_in, H = 768, 50
        self.mode = mode
        num_classes = 1 if mode == 'regression' else 3

        self.bert = AutoModelForSequenceClassification.from_pretrained('pborchert/BusinessBERT', num_labels=num_classes)

        self.multi_label_classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, 10),
            nn.Sigmoid()
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]
        cls_token_embedding = last_hidden_state[:, 0, :]

        if self.mode == 'regression':
            return outputs.logits, self.multi_label_classifier(cls_token_embedding)
        elif self.mode == 'multiclass':
            return outputs.logits, self.multi_label_classifier(cls_token_embedding)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("pborchert/BusinessBERT")

# Load and preprocess the dataset
ogpath = "combined.csv"
dataset = load_dataset('csv', data_files={'train': "train_" + ogpath, 'test': "test_" + ogpath})

def tokenize_function(examples):
    return tokenizer(examples["paragraph"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = torch.tensor(item['input_ids'])
        attention_mask = torch.tensor(item['attention_mask'])
        regression_label = torch.tensor(item['label'], dtype=torch.float)
        multi_labels = torch.tensor([
            item["scarcity"], item["nonuniform_progress"], item["performance_constraints"],
            item["user_heterogeneity"], item["cognitive"], item["external"], 
            item["internal"], item["coordination"], item["technical"], item["demand"]
        ], dtype=torch.float)
        return input_ids, attention_mask, regression_label, multi_labels

train_data = CustomDataset(train_dataset)
test_data = CustomDataset(test_dataset)

train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_dataloader, config, val_dataloader=None, epochs=3):
    optimizer = AdamW(model.parameters(), lr=config["lr"], eps=config["eps"])
    total_steps = len(train_dataloader) * epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    mse_loss_fn = nn.MSELoss()
    cross_entropy_loss_fn = nn.CrossEntropyLoss()
    multi_label_loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_attn_mask, b_regression_labels, b_multi_labels = [t.to(device) for t in batch]
            model.zero_grad()
            regression_logits, multi_label_logits = model(b_input_ids, b_attn_mask)
            multi_label_loss = multi_label_loss_fn(multi_label_logits, b_multi_labels)
            variable_loss = mse_loss_fn(regression_logits.view(-1, 1), b_regression_labels)
            loss = variable_loss + multi_label_loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_dataloader)
        wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})

def evaluate(model, val_dataloader):
    model.eval()
    total_loss = 0
    mse_loss_fn = nn.MSELoss()
    multi_label_loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in val_dataloader:
            b_input_ids, b_attn_mask, b_regression_labels, b_multi_labels = [t.to(device) for t in batch]
            regression_logits, multi_label_logits = model(b_input_ids, b_attn_mask)
            multi_label_loss = multi_label_loss_fn(multi_label_logits, b_multi_labels)
            variable_loss = mse_loss_fn(regression_logits.view(-1, 1), b_regression_labels)
            loss = variable_loss + multi_label_loss
            total_loss += loss.item()

    avg_loss = total_loss / len(val_dataloader)
    wandb.log({"val_loss": avg_loss})
    print(f"Validation Loss: {avg_loss:.4f}")

# Define a WandB sweep configuration
sweep_config = {
    "method": "grid",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "lr": {"values": [1e-5, 3e-5, 5e-5]},
        "eps": {"values": [1e-6, 1e-8, 1e-10]},
        "batch_size": {"values": [16, 32]}
    },
}

sweep_id = wandb.sweep(sweep_config, project="bert-hyperparameter-search")

# Sweep function
def sweep_train():
    config = wandb.config
    # Ensure default values for 'lr' and 'eps'
    lr = config.get("lr", 3e-5)  # Default value
    eps = config.get("eps", 1e-8)  # Default value
    
    model = BertClassifier(mode=default_mode, freeze_bert=False).to(device)
    train(model, train_dataloader, {"lr": lr, "eps": eps}, val_dataloader=test_dataloader, epochs=3)


wandb.agent(sweep_id, function=sweep_train)
