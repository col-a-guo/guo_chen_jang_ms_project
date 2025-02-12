import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from datasets import load_dataset
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
import torchmetrics
import optuna
from sklearn.metrics import confusion_matrix, classification_report
import random
import numpy as np

seed_value = 1
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # If using CUDA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

version_list = ["bert-uncased", "businessBERT", "colaguo-working"]  # Updated version list

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
            input_ids, attention_mask, features, labels = [t.to(device) for t in batch]
            logits = model(input_ids, attention_mask, features)
            preds = torch.argmax(logits, dim=1)  # Multi-class prediction
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays for sklearn functions
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Generate sklearn classification report
    report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_classes)], zero_division=0) # Added zero_division
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    cm_report = "\nConfusion Matrix:\n"
    cm_report += "            Predicted\n"
    cm_report += "           " + "    ".join(map(str, range(num_classes))) + "\n"
    cm_report += "Actual\n"
    for i, row in enumerate(cm):
        cm_report += f"      {i}   " + "    ".join(map(str, row)) + "\n"
    

    final_report = f"""
Classification Report (Version: {version}, Epoch {epoch if epoch is not None else 'Final'}):\n
{report}\n
{cm_report}
"""


    print(final_report)
    with open("classification_report.txt", "a") as f:
        f.write(final_report + "\n")
    
    f1 = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_classes)], output_dict=True, zero_division=0)['macro avg']['f1-score'] # Added zero_division

    return f1

# Define the model architecture (using global pooling for all versions)
class BertClassifier(nn.Module):
    def __init__(self, version, num_labels=1, freeze_bert=False):
        super(BertClassifier, self).__init__()

        if version == "bert-uncased":
            self.bert = AutoModel.from_pretrained('google-bert/bert-base-uncased')
        elif version == "businessBERT":
            self.bert = AutoModel.from_pretrained('pborchert/BusinessBERT')
        elif version == "colaguo-working":
            self.bert = AutoModel.from_pretrained('colaguo/working')
        else:
           raise ValueError(f"Invalid model version: {version}")
        
        self.version = version  # Store the version
        
        self.linear_features = nn.Sequential(
            nn.Linear(11, 8),
            nn.ReLU()
        )

        self.cls_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU()
        )

        self.final_classifier = nn.Linear(128 + 8, num_labels)
        # more or less linear layers
        # linear 128 -> num_labels

        self.pooling = nn.AdaptiveAvgPool1d(1) # Global average pooling layer


        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    #TODO: Add bottleneck features here
                #feedforward, sequential, 11 -> 8 -> num_labels, concatenate with pooled
                #try simple concatenate, then try lower weight/layer down to 128, 64, etc
                #try business, our bert, hybridization, ??
                #focus on / find tokens with captum?
                #check library for past reports maybe
                
    def forward(self, input_ids, attention_mask, features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Global average pooling
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.pooling(last_hidden_state.permute(0, 2, 1)).squeeze(-1)
        
        bert_output = self.cls_head(pooled_output)

        linear_features_output = self.linear_features(features)
        
        combined_output = torch.cat((bert_output, linear_features_output), dim=1)

        logits = self.final_classifier(combined_output)
        return logits



# Function to load the correct tokenizer
def load_tokenizer(version):
    if version == "bert-uncased":
        return AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    elif version == "businessBERT":
        return AutoTokenizer.from_pretrained('pborchert/BusinessBERT')
    elif version == "colaguo-working":
        return AutoTokenizer.from_pretrained('colaguo/bottleneckBERT')
    else:
        raise ValueError(f"Invalid model version: {version}")

# Load dataset and preprocess
ogpath = "feb_6_stitched.csv"
dataset = load_dataset('csv', data_files={'train': "train_" + ogpath, 'test': "test_" + ogpath})

# Truncate dataset; useful to avoid resampling errors due to requesting more samples than exist
# also reducing to very small numbers for testing
def truncate_dataset(dataset):
    k = round(len(dataset)*0.97)
    random_indices = random.sample(range(len(dataset)), k)
    return dataset.select(random_indices)

dataset = {k: truncate_dataset(v) for k, v in dataset.items()}


# Filter out label 2
def filter_label_2(dataset):
    filtered_dataset = dataset.filter(lambda example: example['label'] != 2)
    return filtered_dataset

dataset = {k: filter_label_2(v) for k, v in dataset.items()}

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["paragraph"], padding="max_length", truncation=True, max_length=512)

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
        features = torch.tensor([item['scarcity'], item['nonuniform_progress'], item['performance_constraints'], item['user_heterogeneity'], item['cognitive'], item['external'], item['internal'], item['coordination'], item['transactional'], item['technical'], item['demand']], dtype=torch.float)
        return input_ids, attention_mask, features, label

# Training function
def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, loss_fn, patience=7, num_classes=2, version=None):
    model.to(device)
    best_f1 = 0.0  
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids, attention_mask, features, labels = [t.to(device) for t in batch]
            model.zero_grad()
            logits = model(input_ids, attention_mask, features)
            loss = loss_fn(logits, labels) # Weighted CrossEntropyLoss
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
                input_ids, attention_mask, features, labels = [t.to(device) for t in batch]
                logits = model(input_ids, attention_mask, features)
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

    return best_f1 

# Optuna hyperparameter optimization
def objective(trial, version, train_data, test_data, loss_fn):
    lr = trial.suggest_loguniform("lr", default_lr, default_lr) # Use defaults
    eps = trial.suggest_loguniform("eps", default_eps, default_eps) # Use defaults
    batch_size = trial.suggest_categorical("batch_size", [16, 32]) # Use defaults


    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = BertClassifier(version, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps)
    total_steps = len(train_dataloader) * 20
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    val_f1 = train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, epochs=20, loss_fn=loss_fn, version=version)
    with open("classification_report.txt", "a") as f:
        f.write(f"Run Parameters for {version}:\n lr: {lr}, eps: {eps}, batch_size: {batch_size}\n\n")
    return -val_f1 # Optuna minimizes, we want to maximize F1 so return negative F1


# Main loop
for version in version_list:
    print(f"\n----- Running with {version} -----")

    tokenizer = load_tokenizer(version)
    tokenized_datasets = {split: data.map(lambda examples: tokenize_function(examples, tokenizer), batched=True) for split, data in dataset.items()}
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    train_data = CustomDataset(train_dataset)
    test_data = CustomDataset(test_dataset)

    # Undersampling to balance labels
    train_labels = [item['label'] for item in train_dataset]
    label_counts = Counter(train_labels)
    print("Original label distribution:", label_counts)

    # Determine the minimum count of a class
    min_count = min(label_counts.values())
    
    # Apply undersampling to the training data
    sampler = RandomUnderSampler(sampling_strategy={0: min_count*3, 1: min_count}) #3200:400
    train_indices = list(range(len(train_labels)))
    resampled_indices, resampled_labels = sampler.fit_resample(np.array(train_indices).reshape(-1, 1), np.array(train_labels))
    resampled_indices = resampled_indices.flatten().tolist()
    
    resampled_train_data = torch.utils.data.Subset(train_data, resampled_indices)
    resampled_label_counts = Counter(resampled_labels)
    print("Resampled label distribution:", resampled_label_counts)


    normalized_weights = torch.tensor([0.4, 1.0])
    loss_fn = nn.CrossEntropyLoss(weight=normalized_weights.to(device))
    #Disable reweighting for now
    # # Calculate Class Weights
    # class_weights = torch.tensor([1.0 / count for count in label_counts.values()], dtype=torch.float)
    # #weights based on maximum of 
    # max_weight = max(class_weights)
    # min_weight = min(class_weights)
    # geom_mean = (max_weight*min_weight)**0.5 #min was bad, max was bad, trying geometric mean
    # normalized_weights = class_weights / geom_mean 
    
    # Initialize Model, Print Initial Weights
    model = BertClassifier(version, num_labels=2).to(device) # Initialize before weights
    print(f"\n----- Initial Weights for {version} -----")
    print("\nBERT Initial Weights:")
    for name, param in model.bert.named_parameters():
      if "weight" in name: # Only print weights
        print(f"  {name}: {param.data}")
    
    print("\nClassification Head Initial Weights:")
    for name, param in model.cls_head.named_parameters():
      if "weight" in name:
        print(f"  {name}: {param.data}")

    # Optuna Hyperparameter Tuning with reduced trials
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, version, resampled_train_data, test_data, loss_fn), n_trials=2) # Reduced Trials
    print("Best hyperparameters:", study.best_params)

    # Final evaluation
    model = BertClassifier(version, num_labels=2).to(device)  # Update num_labels to match dataset
    train_dataloader = DataLoader(resampled_train_data, batch_size=study.best_params['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=study.best_params['batch_size'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=study.best_params['lr'], eps=study.best_params['eps'])
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 10)
    train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, epochs=20, loss_fn=loss_fn, num_classes=2, version=version)
    generate_classification_report(model, test_dataloader, num_classes=2, version=version)