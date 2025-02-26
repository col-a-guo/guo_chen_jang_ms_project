import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import get_scheduler
from datasets import load_dataset
from collections import Counter
from imblearn.over_sampling import RandomOverSampler  # Changed import
import torchmetrics
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import random
import numpy as np
from huggingface_hub import PyTorchModelHubMixin
import torch.nn.functional as F
import pandas as pd

seed_value = 1
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # If using CUDA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

version_list = ["colaguo-working","businessBERT",  "bert-uncased"]  # Updated version list

# Default hyperparameters (No Optuna anymore)
default_lr = 1.141204543279205e-05
default_eps = 6.748313060587885e-08
default_batch_size = 32
default_dropout_rate = 0.1  # Added default dropout

# Define prediction thresholds globally
THRESHOLD_0 = 0.7
THRESHOLD_1 = 1.6


# Function to generate regression report
def generate_regression_report(model, dataloader, epoch=None, version=None):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, features, labels = [t.to(device) for t in batch]
            predictions = model(input_ids, attention_mask, features).squeeze()  # Regression output
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    report = f"""
Regression Report (Version: {version}, Epoch {epoch if epoch is not None else 'Final'}):
Mean Squared Error: {mse:.4f}
R-squared: {r2:.4f}
"""

    print(report)
    with open("regression_report.txt", "a") as f:
        f.write(report + "\n")

    return r2  # Return R-squared for early stopping


# Function to generate classification report from regression output using custom thresholds
def generate_classification_report_from_regression(model, dataloader, num_classes=3, epoch=None, version=None):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, features, labels = [t.to(device) for t in batch]
            predictions = model(input_ids, attention_mask, features).squeeze()  # Regression output

            # Apply custom thresholds
            predicted_classes = torch.zeros_like(predictions, dtype=torch.int)
            predicted_classes[predictions <= THRESHOLD_0] = 0
            predicted_classes[(predictions > THRESHOLD_0) & (predictions <= THRESHOLD_1)] = 1
            predicted_classes[predictions > THRESHOLD_1] = 2

            all_preds.extend(predicted_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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
Classification Report (from Regression, Version: {version}, Epoch {epoch if epoch is not None else 'Final'}):\n
{report}\n
{cm_report}
"""

    print(final_report)
    with open("classification_report_from_regression.txt", "a") as f:
        f.write(final_report + "\n")

# Define the model architecture (using global pooling for all versions)
class BertClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(self, version, num_labels=3, freeze_bert=False, dropout_rate=0.1): # Changed num_labels to 3 and added dropout_rate
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
        self.dropout_rate = dropout_rate  # Store the dropout rate
        
        self.linear_features = nn.Sequential(
            nn.Linear(11, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate) # Added dropout
        )

        self.cls_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate) # Added dropout
        )

        self.linear_combined_layer = nn.Sequential(
            nn.Linear(128 + 16, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate) # Added dropout
            )
        
        self.final_regressor = nn.Linear(16, 1)  # Output is 1 for regression

        self.pooling = nn.AdaptiveAvgPool1d(1) # Global average pooling layer


        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Global average pooling
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.pooling(last_hidden_state.permute(0, 2, 1)).squeeze(-1)
        
        bert_output = self.cls_head(pooled_output)

        linear_features_output = self.linear_features(features)
        
        combined_output = torch.cat((bert_output, linear_features_output), dim=1)

        linear_layer_output = self.linear_combined_layer(combined_output)

        prediction = self.final_regressor(linear_layer_output) #No softmax or sigmoid

        # Apply 2*sigmoid activation
        prediction = 2 * torch.sigmoid(prediction)

        return prediction


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
ogpath = "feb_24_combined.csv"
dataset = load_dataset('csv', data_files={'train': "train_" + ogpath, 'test': "test_" + ogpath})

# Truncate dataset; useful to avoid resampling errors due to requesting more samples than exist
# also reducing to very small numbers for testing
def truncate_dataset(dataset):
    k = round(len(dataset)*0.99)
    random_indices = random.sample(range(len(dataset)), k)
    return dataset.select(random_indices)

dataset = {k: truncate_dataset(v) for k, v in dataset.items()}


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
        # CHANGED LABEL DTYPE TO FLOAT! IMPORTANT!
        label = torch.tensor(item['label'], dtype=torch.float)  
        features = torch.tensor([item['scarcity'], item['nonuniform_progress'], item['performance_constraints'], item['user_heterogeneity'], item['cognitive'], item['external'], item['internal'], item['coordination'], item['transactional'], item['technical'], item['demand']], dtype=torch.float)
        return input_ids, attention_mask, features, label

# Training function
def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, loss_fn, patience=5, version=None, lr=default_lr, eps=default_eps, batch_size=default_batch_size, dropout_rate=default_dropout_rate):
    model.to(device)
    best_r2 = -float('inf')  # Initialize with negative infinity for maximization
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids, attention_mask, features, labels = [t.to(device) for t in batch]
            model.zero_grad()
            predictions = model(input_ids, attention_mask, features).squeeze()  # Regression output
            loss = loss_fn(predictions, labels)
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
                predictions = model(input_ids, attention_mask, features).squeeze()
                val_loss += loss_fn(predictions, labels).item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        
        # Generate and save the regression report every epoch
        r2_score_val = generate_regression_report(model, val_dataloader, epoch=epoch+1, version=version)

        # Generate and save the classification report from regression output every epoch
        generate_classification_report_from_regression(model, val_dataloader, num_classes=3, epoch=epoch+1, version=version)


        # Early stopping based on R-squared
        if r2_score_val > best_r2:
            best_r2 = r2_score_val
            patience_counter = 0
            print(f"New best R2: {best_r2}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    tokenizer.push_to_hub("colaguo/my-awesome-model")
    # push to the hub
    model.push_to_hub("colaguo/my-awesome-model")
    return best_r2


# Oversampling Function
def oversample_dataset(dataset):
    """Oversamples the dataset to balance the occurrences of labels,
    but only considers and oversamples 0.0, 1.0, and 2.0."""

    # Convert dataset to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()

    # Filter the DataFrame to include only rows where the label is 0.0, 1.0, or 2.0
    df_filtered = df[df['label'].isin([0.0, 1.0, 2.0])]

    # Separate features and labels from the filtered DataFrame
    X = df_filtered.drop('label', axis=1)
    y = df_filtered['label']

    # Initialize RandomOverSampler
    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=seed_value)

    # Fit and apply oversampling to the training data
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Create a new DataFrame from the resampled data
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df['label'] = y_resampled

    # Concatenate the original DataFrame with the oversampled DataFrame
    # First, exclude the filtered rows from the original DataFrame
    df_excluded = df[~df['label'].isin([0.0, 1.0, 2.0])]

    # Concatenate the excluded rows with the resampled data
    final_df = pd.concat([df_excluded, resampled_df], ignore_index=True)

    # Shuffle the DataFrame to mix the original and oversampled data
    final_df = final_df.sample(frac=1, random_state=seed_value).reset_index(drop=True)

    # Convert back to Hugging Face Dataset
    resampled_dataset = dataset.from_pandas(final_df)
    resampled_dataset.set_format(type=dataset.format["type"], columns=dataset.column_names)

    print(f"Oversampled dataset. Original size: {len(dataset)}, New size: {len(resampled_dataset)}")
    return resampled_dataset

# Main loop
for version in version_list:
    print(f"\n----- Running with {version} -----")

    tokenizer = load_tokenizer(version)
    tokenized_datasets = {}
    for split, data in dataset.items():
        # APPLY OVERSAMPLING BEFORE TOKENIZATION
        if split == "train":
            data = oversample_dataset(data)

        tokenized_datasets[split] = data.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)

    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    train_data = CustomDataset(train_dataset)
    test_data = CustomDataset(test_dataset)

    # Oversampling to balance labels
    train_labels = [item['label'] for item in train_dataset]
    label_counts = Counter(train_labels)
    print("Original label distribution:", label_counts)

    # Determine the maximum count of a class
    max_count = max(label_counts.values())

    # Apply oversampling to the training data
    sampling_strategy = {}
    for i in range(3):  # Iterate through all possible labels
        if i in label_counts:
            sampling_strategy[i] = max_count  # Oversample to the maximum count
        else:
            sampling_strategy[i] = max_count  # If class not present, create enough to oversample it.
    oversampler = RandomOverSampler(sampling_strategy=sampling_strategy)

    train_indices = list(range(len(train_labels)))
    resampled_indices, resampled_labels = oversampler.fit_resample(np.array(train_indices).reshape(-1, 1), np.array(train_labels))
    resampled_indices = resampled_indices.flatten().tolist()

    # Create resampled Subset
    resampled_train_data = Subset(train_data, resampled_indices)


    resampled_label_counts = Counter(resampled_labels)
    print("Resampled label distribution:", resampled_label_counts)


    # Adjust normalized weights to account for 3 labels
    normalized_weights = torch.tensor([1.0, 1.0, 2.0]) # set all weights to 1 initially
    loss_fn = nn.CrossEntropyLoss(weight=normalized_weights.to(device))
    
    # Initialize Model, Print Initial Weights
    model = BertRegressor(version).to(device) # Initialize before weights

    # Training loop with default parameters
    train_dataloader = DataLoader(train_data, batch_size=default_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=default_batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=default_lr, eps=default_eps)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 10)
    loss_fn = nn.MSELoss() #Use MSE Loss

    train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, epochs=40, loss_fn=loss_fn, version=version, lr=default_lr, eps=default_eps, batch_size=default_batch_size, dropout_rate=default_dropout_rate)
    generate_regression_report(model, test_dataloader, version=version)
    generate_classification_report_from_regression(model, test_dataloader, num_classes=3, version=version)