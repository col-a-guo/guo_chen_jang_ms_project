import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from datasets import load_dataset
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

seed_value = 1
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # If using CUDA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

version_list = ['businessBERT',"bottleneckBERT","bert-uncased"]  # Updated version list

# Default hyperparameters (removed Optuna dependency)
default_lr = 5e-5 #initial learning rate
default_eps = 6.748313060587885e-08
default_batch_size = 32
num_epochs = 20
patience = 4 #For early stopping
target_lr = 8e-6 #Target after 10 epochs
warmup_proportion = 0.2

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
    report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_classes)], digits=4, zero_division=0) # Added zero_division
    
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
        elif version == "bottleneckBERT":
            self.bert = AutoModel.from_pretrained('colaguo/bottleneckBERTlarge')
        else:
           raise ValueError(f"Invalid model version: {version}")
        
        self.version = version  # Store the version
        
        self.linear_features = nn.Sequential(
            nn.Linear(11, 16),
            nn.ReLU()
        )

        self.linear_bottid = nn.Sequential(
            nn.Linear(num_bottid_categories, 16),  # Linear layer for bottid encoding
            nn.ReLU()
        )

        self.cls_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU()
        )

        self.linear_combined_layer = nn.Sequential(
            nn.Linear(128 + 16 + 16, 32), #Concatenate additional features here
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
    elif version == "bottleneckBERT":
        return AutoTokenizer.from_pretrained('pborchert/BusinessBERT')
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

# Training function
def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, warmup_scheduler, decay_scheduler, epochs, loss_fn, patience=4, num_classes=2, version=None):
    model.to(device)
    best_f1 = 0.0
    patience_counter = 0
    current_step = 0
    best_epoch = 0  # Keep track of the epoch with the best F1
    output_dir = "model_output"  # Define directory
    best_model_state = None #To store the state dict of best model

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)



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

            if current_step < warmup_steps:
                warmup_scheduler.step()
            decay_scheduler.step()  # Always step the decay scheduler

            current_step += 1
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
            best_epoch = epoch + 1 # Store the best epoch
            patience_counter = 0
            best_model_state = model.state_dict() # Save best model state
            print(f"New best F1 score: {best_f1:.4f} at epoch {epoch+1}.") # Epoch logging
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model weights, then save
    if best_model_state is not None:
        model.load_state_dict(best_model_state) # load best model
        model_filename = f"model_output/model_version_{version}.pth"  # Added version and timestamp
        torch.save(model.state_dict(), model_filename)  # Save the model's weights
        print(f"Best model (version {version}) saved to {model_filename} with F1 {best_f1:.4f}")

    
    tokenizer.push_to_hub(f"colaguo/{version}_finetune_feb24")
    # push to the hub
    model.push_to_hub(f"colaguo/{version}_finetune_feb24")

    print(f"Training completed. Best F1 score: {best_f1:.4f} achieved at epoch {best_epoch}.") #Log the best F1 after training.
    return best_f1

# Main loop
for version in version_list:
    print(f"\n----- Running with {version} -----")

    tokenizer = load_tokenizer(version)
    tokenized_datasets = {split: data.map(lambda examples: tokenize_function(examples, tokenizer), batched=True) for split, data in dataset.items()}
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]

    num_bottid_categories = train_encoded.shape[1] #Determine the number of bottid categories
    train_data = CustomDataset(train_dataset, bottid_categories=num_bottid_categories) # pass to CustomDataset
    test_data = CustomDataset(test_dataset, bottid_categories=num_bottid_categories) # pass to CustomDataset

    # Undersampling to balance labels
    train_labels = [item['label'] for item in train_dataset]
    label_counts = Counter(train_labels)
    print("Original label distribution:", label_counts)

    # Determine the minimum count of a class
    min_count = min(label_counts.values())
    
    # # Apply undersampling to the training data
    # sampler = RandomUnderSampler(sampling_strategy={0:int(round(min_count*1.4)), 1:min_count}) #3200:400
    # train_indices = list(range(len(train_labels)))
    # resampled_indices, resampled_labels = sampler.fit_resample(np.array(train_indices).reshape(-1, 1), np.array(train_indices))
    # resampled_indices = resampled_indices.flatten().tolist()
    
    # resampled_train_data = torch.utils.data.Subset(train_data, resampled_indices)
    # resampled_label_counts = Counter(resampled_labels)
    # print("Resampled label distribution:", resampled_label_counts)

    #Use full dataset
    train_data_loader = DataLoader(train_data, batch_size=default_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=default_batch_size)


    normalized_weights = torch.tensor([1.0, 1.2])
    loss_fn = nn.CrossEntropyLoss(weight=normalized_weights.to(device))
    
    # Initialize Model
    model = BertClassifier(version, num_labels=2, num_bottid_categories=num_bottid_categories).to(device) # Initialize before weights, pass num_bottid_categories here

    #train_dataloader = DataLoader(resampled_train_data, batch_size=default_batch_size, shuffle=True)
    #Remove train_dataloader and just use train_data instead to use full dataloaders
    train_dataloader = train_data_loader
    test_dataloader = DataLoader(test_data, batch_size=default_batch_size)

    #Set up the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=default_lr, eps=default_eps)

    #Calculate warmup steps based on epochs
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(warmup_proportion * total_steps)

    #Get warmup + decay schedulers
    warmup_scheduler, decay_scheduler = get_exponential_warmup_schedule(
        optimizer,
        warmup_steps,
        default_lr,
        target_lr,
        num_epochs,
        total_steps
    )

    #Train and evaluate
    train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, warmup_scheduler, decay_scheduler, epochs=num_epochs, loss_fn=loss_fn, num_classes=2, version=version, patience=patience)
    generate_classification_report(model, test_dataloader, num_classes=2, version=version)