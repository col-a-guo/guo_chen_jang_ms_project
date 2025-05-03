import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from datasets import load_dataset
from datasets import Dataset as HFDataset #Import Dataset
from collections import Counter
#from imblearn.under_sampling import RandomUnderSampler #No longer import undersampler
import torchmetrics
from sklearn.metrics import confusion_matrix, classification_report
import random
import numpy as np
from huggingface_hub import PyTorchModelHubMixin
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
import os  # Import the 'os' module
from datetime import datetime # Import the 'datetime' module

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# Load dataset
combined = pd.read_csv("feb_24_combined.csv")

# Define features (including 'label' as it's needed before separating X and y)
features = [
    "scarcity", "nonuniform_progress", "performance_constraints", "user_heterogeneity",
    "cognitive", "external", "internal", "coordination", "transactional", "technical",
    "demand", "label", "bottid", "word_count",
]

# Control variables (Note: word_count used in quad features. Could also use length_approx, source)
control_vars = []


# Remove rows where label is 2.0
combined = combined[combined["label"] != 2.0]

# One-Hot Encode 'bottid'
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
bottid_encoded = encoder.fit_transform(combined[['bottid']])
bottid_df = pd.DataFrame(bottid_encoded, columns=encoder.get_feature_names_out(['bottid']))
combined = pd.concat([combined.reset_index(drop=True), bottid_df], axis=1)
combined = combined.drop('bottid', axis=1)  # Drop original 'bottid' column

# Prepare data - determine the feature list after one-hot encoding
all_features = [
    "scarcity", "nonuniform_progress", "performance_constraints", "user_heterogeneity",
    "cognitive", "external", "internal", "coordination", "transactional", "technical",
    "demand"
] + list(bottid_df.columns)  # Add one-hot encoded column names

# Ensure all features are present in the data; otherwise, add with default 0
for feature in all_features:
    if feature not in combined.columns:
        combined[feature] = 0

#minmax scale
def prepare_X_data(data, all_features):
    X = data[all_features]
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X = prepare_X_data(combined, all_features)  # Pass all features after one-hot encoding
y = combined["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the estimator
estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42))]

clf = RandomForestClassifier(n_estimators=10, random_state=42)

# Function to calculate permutation importance
def calculate_permutation_importance(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    result = permutation_importance(clf, X_train, y_train, n_repeats=8, random_state=42)
    return result.importances_mean

# Function to print classification report
def print_classification_report(clf, X_test, y_test, dataset_name):
    y_pred = clf.predict(X_test)
    print(f"Classification Report for {dataset_name}:")
    print(classification_report(y_test, y_pred))


# Train and evaluate the model
clf.fit(X_train, y_train)
print_classification_report(clf, X_test, y_test, "Combined")

# Permutation importance
perm_importance = calculate_permutation_importance(clf, X_train, y_train)

# Set importance of 'length_approx' to 0 in all plots (to avoid errors)
# Extract feature names from the processed data, since one-hot encoding changes them
labels = list(X.columns)

# Normalize the permutation importance values (scale each to 0-1 range)
perm_importance = perm_importance / np.max(perm_importance)

# --- Feature Selection ---
# Get the indices of the top 12 features
top_feature_indices = perm_importance.argsort()[-12:][::-1]  # Get indices of top 12
selected_features = list(X.columns[top_feature_indices])  # Get names of top 12 features

print("Selected Features:", selected_features)


#############################START BERT MODEL CODE#############################

#Note: Almost identical to BERTearnings.py, so comments are sparse

seed_value = 1
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # If using CUDA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

version_list = ["bert-uncased", "businessBERT","bottleneckBERT"]  # Updated version list

# Default hyperparameters (removed Optuna dependency)
default_lr = 5e-5 #initial learning rate
default_eps = 6.748313060587885e-08
default_batch_size = 32
num_epochs = 20
patience = 4 
target_lr = 8e-6 #Target after 10 epochs
warmup_proportion = 0.2

# Function to generate classification report for multi-class
def generate_classification_report(model, dataloader, num_classes, epoch=None, version=None):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, bottid_encoded, labels = [t.to(device) for t in batch] #Unpack bottid
            logits = model(input_ids, attention_mask, bottid_encoded) # Pass bottid to model
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

class BertClassifier(nn.Module, PyTorchModelHubMixin):
# Define the model architecture (using global pooling for all versions)class BertClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(self, version, num_labels=1, freeze_bert=False, selected_feature_count=12): # ADDED: selected_feature_count
        super(BertClassifier, self).__init__()

        if version == "bert-uncased":
            self.bert = AutoModel.from_pretrained('google-bert/bert-base-uncased')
        elif version == "businessBERT":
            self.bert = AutoModel.from_pretrained('pborchert/BusinessBERT')
        elif version == "bottleneckBERT":
            self.bert = AutoModel.from_pretrained('colaguo/bottleneckBERTsmall')
        else:
           raise ValueError(f"Invalid model version: {version}")

        self.version = version

        self.linear_features = nn.Sequential( 
            nn.Linear(selected_feature_count, 16),
            nn.ReLU()
        ) #use selected features -> 16 instead of key + bottid -> 16 + 8

        self.cls_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU()
        )

        self.linear_combined_layer = nn.Sequential(
            nn.Linear(128 + 16, 32),
            nn.ReLU()
        )

        self.final_classifier = nn.Linear(32, num_labels)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, selected_features): 
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.pooling(last_hidden_state.permute(0, 2, 1)).squeeze(-1)
        bert_output = self.cls_head(pooled_output)

        linear_features_output = self.linear_features(selected_features) 

        combined_output = torch.cat((bert_output, linear_features_output), dim=1) 
        linear_layer_output = self.linear_combined_layer(combined_output)
        logits = self.final_classifier(linear_layer_output)
        return logits
def load_tokenizer(version):
    if version == "bert-uncased":
        return AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    elif version == "businessBERT":
        return AutoTokenizer.from_pretrained('pborchert/BusinessBERT')
    elif version == "bottleneckBERT":
        return AutoTokenizer.from_pretrained('pborchert/BusinessBERT')
    else:
        raise ValueError(f"Invalid model version: {version}")

# Load dataset and preprocess
ogpath = "feb_24_combined.csv"
ogpath2 = "feb_20_stitched.csv"

# Load the datasets
dataset1 = load_dataset('csv', data_files={'train': "train_" + ogpath, 'test': "test_" + ogpath})
dataset2 = load_dataset('csv', data_files={'train': "train_" + ogpath2, 'test': "test_" + ogpath2})

# Load the CSVs into pandas to encode bottid correctly, then pass back into the HF dataset.
train_df1 = pd.read_csv("train_" + ogpath)
test_df1 = pd.read_csv("test_" + ogpath)
train_df2 = pd.read_csv("train_" + ogpath2)
test_df2 = pd.read_csv("test_" + ogpath2)

# Combine the dataframes
train_df = pd.concat([train_df1, train_df2], ignore_index=True)
test_df = pd.concat([test_df1, test_df2], ignore_index=True)

encoder = OneHotEncoder(handle_unknown='ignore')

encoder.fit(train_df[['bottid']])

train_encoded = encoder.transform(train_df[['bottid']]).toarray()
test_encoded = encoder.transform(test_df[['bottid']]).toarray()

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
# Create HF datasets from the pandas DataFrames.
hf_train_dataset = HFDataset.from_pandas(train_df)
hf_test_dataset = HFDataset.from_pandas(test_df)

# Combine the train and test datasets into a single dataset dictionary
dataset = {
    'train': hf_train_dataset,
    'test': hf_test_dataset
}

# Truncate dataset; useful to avoid resampling errors due to requesting more samples than exist
# also reducing to very small numbers for testing
def truncate_dataset(dataset):
    k = round(len(dataset)*0.99)
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


# Learning Rate Scheduler: Exponential Decay with Linear Warmup
def get_exponential_warmup_schedule(optimizer, warmup_steps, initial_lr, target_lr, num_epochs, total_steps):
    """
    Combines a linear warmup with an exponential decay to reach a target learning rate
    after a specified number of epochs.

    Args:
        optimizer: The optimizer.
        warmup_steps: Number of steps for the warmup phase.
        initial_lr: The initial learning rate.
        target_lr: The target learning rate after num_epochs.
        num_epochs: The number of epochs to reach the target_lr.
        total_steps: Total number of training steps.

    Returns:
        A tuple of learning rate schedulers (warmup, exponential).
    """

    def warmup_lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0  # Keep LR at 1.0 after warmup

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)

    # Calculate decay rate to reach target_lr after num_epochs
    decay_rate = (target_lr / initial_lr)**(1 / (total_steps - warmup_steps))
    decay_scheduler = ExponentialLR(optimizer, gamma=decay_rate)

    return warmup_scheduler, decay_scheduler


class CustomDataset(Dataset):
    def __init__(self, dataset, selected_features, bottid_categories=29): # ADDED: selected_features
        self.dataset = dataset
        self.selected_features = selected_features  # Store selected feature names
        self.bottid_categories = bottid_categories

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = torch.tensor(item['input_ids'])
        attention_mask = torch.tensor(item['attention_mask'])
        label = torch.tensor(item['label'], dtype=torch.long)

        # Extract and combine the selected features into a single tensor
        selected_feature_values = []
        for feature_name in self.selected_features:
            selected_feature_values.append(item[feature_name]) 
        
        selected_features = torch.tensor(selected_feature_values, dtype=torch.float) 

        return input_ids, attention_mask, selected_features, label

# --- Training Loop ---
for version in version_list:
    print(f"\n----- Running with {version} -----")

    tokenizer = load_tokenizer(version)
    tokenized_datasets = {split: data.map(lambda examples: tokenize_function(examples, tokenizer), batched=True) for split, data in dataset.items()}
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    num_bottid_categories = train_encoded.shape[1]
    # --- Pass selected_features to CustomDataset ---
    train_data = CustomDataset(train_dataset, selected_features, bottid_categories=num_bottid_categories) 
    test_data = CustomDataset(test_dataset, selected_features, bottid_categories=num_bottid_categories) 

    # (Undersampling and DataLoader setup remain similar)

    normalized_weights = torch.tensor([1.0, 1.2])
    loss_fn = nn.CrossEntropyLoss(weight=normalized_weights.to(device))

    # --- Initialize Model with selected_feature_count ---
    model = BertClassifier(version, num_labels=2, selected_feature_count=len(selected_features)).to(device)

    train_dataloader = DataLoader(train_data, batch_size=default_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=default_batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=default_lr, eps=default_eps)

    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(warmup_proportion * total_steps)

    warmup_scheduler, decay_scheduler = get_exponential_warmup_schedule(
        optimizer,
        warmup_steps,
        default_lr,
        target_lr,
        num_epochs,
        total_steps
    )


    def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, warmup_scheduler, decay_scheduler, epochs, loss_fn, patience=4, num_classes=2, version=None):
        model.to(device)
        best_f1 = 0.0
        patience_counter = 0
        current_step = 0
        best_epoch = 0
        output_dir = "model_output"
        best_model_state = None

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_dataloader:
                input_ids, attention_mask, selected_features, labels = [t.to(device) for t in batch] 
                model.zero_grad()
                logits = model(input_ids, attention_mask, selected_features)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if current_step < warmup_steps:
                    warmup_scheduler.step()
                decay_scheduler.step()

                current_step += 1
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_dataloader)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids, attention_mask, selected_features, labels = [t.to(device) for t in batch] 
                    logits = model(input_ids, attention_mask, selected_features) 
                    val_loss += loss_fn(logits, labels).item()

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

            # Generate and save the classification report every epoch
            f1_score = generate_classification_report(model, val_dataloader, num_classes, epoch=epoch + 1, version=version)

            # Early stopping based on F1 score
            if f1_score > best_f1:
                best_f1 = f1_score
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_state = model.state_dict()
                print(f"New best F1 score: {best_f1:.4f} at epoch {epoch + 1}.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Load best model weights, then save
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model_filename = f"model_output/model_version_{version}.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"Best model (version {version}) saved to {model_filename} with F1 {best_f1:.4f}")

        tokenizer.push_to_hub(f"colaguo/{version}_RF_finetunefeb24")
        model.push_to_hub(f"colaguo/{version}_RF_finetunefeb24")

        print(f"Training completed. Best F1 score: {best_f1:.4f} achieved at epoch {best_epoch}.")
        return best_f1

    #Train and evaluate
    train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, warmup_scheduler, decay_scheduler, epochs=num_epochs, loss_fn=loss_fn, num_classes=2, version=version, patience=patience)
    generate_classification_report(model, test_dataloader, num_classes=2, version=version)