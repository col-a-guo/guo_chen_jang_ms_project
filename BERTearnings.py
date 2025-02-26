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
from huggingface_hub import PyTorchModelHubMixin
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

seed_value = 1
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # If using CUDA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

version_list = ["colaguo-working", "bert-uncased", "businessBERT"]  # Updated version list

# Default hyperparameters for Optuna
default_lr = 3.141204543279205e-05
default_eps = 6.748313060587885e-08
default_batch_size = 32

# Function to generate classification report for multi-class
def generate_classification_report(model, dataloader, num_classes, epoch=None, version=None):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, features, bottid_encoded, labels = [t.to(device) for t in batch] #Unpack bottid
            logits = model(input_ids, attention_mask, features, bottid_encoded) # Pass bottid to model
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
class BertClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(self, version, num_labels=1, freeze_bert=False, num_bottid_categories=29): # Added num_bottid_categories
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
            nn.Linear(11, 16),
            nn.ReLU()
        )

        self.linear_bottid = nn.Sequential(
            nn.Linear(num_bottid_categories, 8),  # Linear layer for bottid encoding
            nn.ReLU()
        )


        self.cls_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU()
        )

        self.linear_combined_layer = nn.Sequential(
            nn.Linear(128 + 16 + 8, 32), #Concatenate additional features here
            nn.ReLU())
        
        self.final_classifier = nn.Linear(32, num_labels)
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
                
    def forward(self, input_ids, attention_mask, features, bottid_encoded): # Take bottid as input
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Global average pooling
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.pooling(last_hidden_state.permute(0, 2, 1)).squeeze(-1)
        
        bert_output = self.cls_head(pooled_output)

        linear_features_output = self.linear_features(features)
        bottid_output = self.linear_bottid(bottid_encoded) # Pass bottid through linear layer


        combined_output = torch.cat((bert_output, linear_features_output, bottid_output), dim=1) #Concatenate bottid


        linear_layer_output = self.linear_combined_layer(combined_output)

        logits = self.final_classifier(linear_layer_output)
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
ogpath = "feb_20_stitched.csv"
dataset = load_dataset('csv', data_files={'train': "train_" + ogpath, 'test': "test_" + ogpath})

# Load the CSVs into pandas to encode bottid correctly, then pass back into the HF dataset.
train_df = pd.read_csv("train_" + ogpath)
test_df = pd.read_csv("test_" + ogpath)

#One-Hot-Encode the bottid features
encoder = OneHotEncoder(handle_unknown='ignore')

encoder.fit(train_df[['bottid']])

train_encoded = encoder.transform(train_df[['bottid']]).toarray()
test_encoded = encoder.transform(test_df[['bottid']]).toarray()

# get_feature_names_out is deprecated, use get_feature_names instead
# but this throws an error locally and I don't want to deal with this
# feature_names = encoder.get_feature_names_out(['bottid'])
feature_names = [f"bottid_{i}" for i in range(train_encoded.shape[1])]

# create a temporary dataframe to store encoded values, with feature names
train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names)
test_encoded_df = pd.DataFrame(test_encoded, columns=feature_names)

#Concatenate new onehot-encoded columns onto original dataframe
train_df = pd.concat([train_df, train_encoded_df], axis=1)
test_df = pd.concat([test_df, test_encoded_df], axis=1)

# Remove the original bottid column
train_df = train_df.drop('bottid', axis=1)
test_df = test_df.drop('bottid', axis=1)

#Convert the dataframes back to HuggingFace datasets
dataset['train'] = dataset['train'].from_pandas(train_df)
dataset['test'] = dataset['test'].from_pandas(test_df)


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
    def __init__(self, dataset, bottid_categories=29): #Added bottid_categories, 29 should be the #. of botIDs
        self.dataset = dataset
        self.bottid_categories = bottid_categories

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = torch.tensor(item['input_ids'])
        attention_mask = torch.tensor(item['attention_mask'])
        label = torch.tensor(item['label'], dtype=torch.long)
        features = torch.tensor([item['scarcity'], item['nonuniform_progress'], item['performance_constraints'], item['user_heterogeneity'], item['cognitive'], item['external'], item['internal'], item['coordination'], item['transactional'], item['technical'], item['demand']], dtype=torch.float)

        # Extract the one-hot encoded bottid features
        bottid_encoded = torch.tensor([item[f"bottid_{i}"] for i in range(self.bottid_categories)], dtype=torch.float)

        return input_ids, attention_mask, features, bottid_encoded, label # Returns bottid encoding, label

# Training function
def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, loss_fn, patience=7, num_classes=2, version=None):
    model.to(device)
    best_f1 = 0.0  
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids, attention_mask, features, bottid_encoded, labels = [t.to(device) for t in batch] #Unpack bottid
            model.zero_grad()
            logits = model(input_ids, attention_mask, features, bottid_encoded) #Pass bottid to model
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
                input_ids, attention_mask, features, bottid_encoded, labels = [t.to(device) for t in batch] #Unpack bottid
                logits = model(input_ids, attention_mask, features, bottid_encoded) #Pass bottid to model
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

    tokenizer.push_to_hub("colaguo/my-awesome-model")
    # push to the hub
    model.push_to_hub("colaguo/my-awesome-model")
    return best_f1 

# Optuna hyperparameter optimization
def objective(trial, version, train_data, test_data, loss_fn, num_bottid_categories=29): #Added num_bottid_categories here
    lr = trial.suggest_loguniform("lr", default_lr, default_lr) # Use defaults
    eps = trial.suggest_loguniform("eps", default_eps, default_eps) # Use defaults
    batch_size = trial.suggest_categorical("batch_size", [16, 32]) # Use defaults


    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = BertClassifier(version, num_labels=2, num_bottid_categories=num_bottid_categories).to(device) # Pass # of bottid categories here
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

    #Regression benefits a lot from oversampling!
    
    # Initialize Model, Print Initial Weights
    model = BertClassifier(version, num_labels=2, num_bottid_categories=num_bottid_categories).to(device) # Initialize before weights, pass num_bottid_categories here

    # Optuna Hyperparameter Tuning with reduced trials
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, version, resampled_train_data, test_data, loss_fn, num_bottid_categories=num_bottid_categories), n_trials=2) # Reduced Trials, pass num_bottid_categories here
    print("Best hyperparameters:", study.best_params)

    # Final evaluation
    model = BertClassifier(version, num_labels=2, num_bottid_categories=num_bottid_categories).to(device)  # Update num_labels to match dataset, pass num_bottid_categories here
    train_dataloader = DataLoader(resampled_train_data, batch_size=study.best_params['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=study.best_params['batch_size'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=study.best_params['lr'], eps=study.best_params['eps'])
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 10)
    train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, epochs=20, loss_fn=loss_fn, num_classes=2, version=version)
    generate_classification_report(model, test_dataloader, num_classes=2, version=version)