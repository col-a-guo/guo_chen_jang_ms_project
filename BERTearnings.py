import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_scheduler
from datasets import load_dataset
import time

default_mode = 'multiclass'
class BertClassifier(nn.Module):
    """Bert Model for Regression, Multi-class and Multi-label Tasks."""
    def __init__(self, mode=default_mode, freeze_bert=False):
        """
        @param mode (str): 'regression' or 'multiclass'. Determines whether to run regression or multi-class classification.
        @param freeze_bert (bool): Set False to fine-tune the BERT model.
        """
        super(BertClassifier, self).__init__()
        D_in, H = 768, 50  # BERT hidden size is 768
        self.mode = mode  # 'regression' or 'multiclass'
        num_classes = 1 if mode == 'regression' else 3  # Default: regression, else multiclass (change as needed)

        # Load the BusinessBERT model
        self.bert = AutoModelForSequenceClassification.from_pretrained('pborchert/BusinessBERT', num_labels=num_classes)

        # Define the multi-label classification head (adjust for your number of labels)
        self.multi_label_classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, 10),  # Adjust number of output labels as necessary
            nn.Sigmoid()
        )

        # Optionally freeze the BERT model layers
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model for both multi-class and multi-label tasks.
        """
        # Pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Extract the hidden states (last hidden state from BERT)
        hidden_states = outputs.hidden_states  # This is a tuple of (layer_0, layer_1, ..., layer_N)
        last_hidden_state = hidden_states[-1]  # The last layer's hidden state
        cls_token_embedding = last_hidden_state[:, 0, :]  # Extract the [CLS] token's embedding

        # Multi-class classification logits (using the logits directly from BERT)
        if self.mode == 'regression':
            regression_logits = outputs.logits
            return regression_logits, self.multi_label_classifier(cls_token_embedding)
        
        elif self.mode == 'multiclass':
            multiclass_logits = outputs.logits  # This will be used for multiclass classification (could modify num_classes)
            return multiclass_logits, self.multi_label_classifier(cls_token_embedding)


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

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Tokenized inputs
        input_ids = torch.tensor(item['input_ids'])
        attention_mask = torch.tensor(item['attention_mask'])

        # Multi-class label (For regression, label is used as continuous, for multiclass as one-hot or categorical)
        regression_label = torch.tensor(item['label'], dtype=torch.float)  # Regression label (continuous)

        # Multi-label features
        multi_labels = torch.tensor([ 
            item["scarcity"],
            item["nonuniform_progress"],
            item["performance_constraints"],
            item["user_heterogeneity"],
            item["cognitive"],
            item["external"],
            item["internal"],
            item["coordination"],
            item["technical"],
            item["demand"]
        ], dtype=torch.float)  # Shape: [10]

        return input_ids, attention_mask, regression_label, multi_labels

train_data = CustomDataset(train_dataset)
test_data = CustomDataset(test_dataset)

train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16)

# Initialize model, optimizer, and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertClassifier(mode=default_mode, freeze_bert=False).to(device)
optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
num_epochs = 3
total_steps = len(train_dataloader) * num_epochs
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
# Global mode variable (toggle between regression and multiclass)
default_mode = 'regression'  # Default to regression, can switch to 'multiclass'

# Define loss functions
mse_loss_fn = nn.MSELoss()  # MSE for regression
cross_entropy_loss_fn = nn.CrossEntropyLoss()  # CrossEntropy for multiclass classification
multi_label_loss_fn = nn.BCEWithLogitsLoss()  # Multi-label Binary Cross Entropy Loss with Logits

# Variable loss function that switches between regression and multiclass
def variable_loss_fn(logits, labels):
    if default_mode == 'regression':
        return mse_loss_fn(logits.view(-1, 1), labels)  # For regression, MSE loss
    elif default_mode == 'multiclass':
        return cross_entropy_loss_fn(logits.view(-1, 3), labels)  # For multiclass, CrossEntropy loss
    else:
        raise ValueError(f"Unsupported mode: {default_mode}")

# Training loop
def train(model, train_dataloader, val_dataloader=None, epochs=4):
    print("Starting training...\n")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'Batch':<8}{'Variable Loss':<12}{'Multilabel Loss':<18}{'Avg Loss':<12}{'Elapsed':<9}")
        print("-" * 60)

        model.train()
        t0_epoch = time.time()
        total_loss = 0
        total_variable_loss = 0
        total_multi_label_loss = 0

        for step, batch in enumerate(train_dataloader):
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_regression_labels, b_multi_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass
            regression_logits, multi_label_logits = model(b_input_ids, b_attn_mask)

            # Compute multi-label loss first (this will always be used)
            multi_label_loss = multi_label_loss_fn(multi_label_logits, b_multi_labels)

            # Use variable_loss_fn to calculate either regression or multiclass loss
            variable_loss = variable_loss_fn(regression_logits, b_regression_labels) if default_mode == 'regression' else variable_loss_fn(multi_label_logits.view(-1, 3), b_multi_labels)

            # Total loss is the sum of variable loss and multi-label loss
            loss = variable_loss + multi_label_loss

            # Accumulate total losses for this batch
            total_loss += loss.item()
            total_variable_loss += variable_loss.item()
            total_multi_label_loss += multi_label_loss.item()

            # Perform a backward pass
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and learning rate
            optimizer.step()
            scheduler.step()

            if step % 10 == 0:
                elapsed = time.time() - t0_epoch
                print(f"{step:<8}{variable_loss.item():<12.6f}{multi_label_loss.item():<18.6f}{loss.item():<12.6f}{elapsed:<9.2f}")

        # Average loss for this epoch
        avg_loss = total_loss / len(train_dataloader) /2
        avg_variable_loss = total_variable_loss / len(train_dataloader)
        avg_multi_label_loss = total_multi_label_loss / len(train_dataloader)

        print(f"\nEpoch {epoch+1} - Average losses:")
        print(f"Variable Loss: {avg_variable_loss:.6f}")
        print(f"Multilabel Loss: {avg_multi_label_loss:.6f}")
        print(f"Avg Loss: {avg_loss:.6f}")

        # Evaluate model on validation data (if provided)
        if val_dataloader:
            evaluate(model, val_dataloader)

# Evaluation function (similar logic as training, dynamic loss computation)
def evaluate(model, val_dataloader):
    model.eval()
    total_loss = 0
    total_variable_loss = 0
    total_multi_label_loss = 0

    with torch.no_grad():
        for batch in val_dataloader:
            b_input_ids, b_attn_mask, b_regression_labels, b_multi_labels = [t.to(device) for t in batch]

            # Perform forward pass
            regression_logits, multi_label_logits = model(b_input_ids, b_attn_mask)

            # Compute multi-label loss first (this will always be used)
            multi_label_loss = multi_label_loss_fn(multi_label_logits, b_multi_labels)

            # Use variable_loss_fn to calculate either regression or multiclass loss
            variable_loss = variable_loss_fn(regression_logits, b_regression_labels) if default_mode == 'regression' else variable_loss_fn(multi_label_logits, b_multi_labels)

            # Total loss is the sum of variable loss and multi-label loss
            loss = variable_loss + multi_label_loss

            # Aggregate total loss
            total_loss += loss.item()
            total_variable_loss += variable_loss.item()
            total_multi_label_loss += multi_label_loss.item()

    avg_loss = total_loss / len(val_dataloader)
    avg_variable_loss = total_variable_loss / len(val_dataloader)
    avg_multi_label_loss = total_multi_label_loss / len(val_dataloader)

    print(f"Validation Results - Loss: {avg_loss:.6f}, Variable Loss: {avg_variable_loss:.6f}, Multi-label Loss: {avg_multi_label_loss:.6f}")

# Start training
train(model, train_dataloader, epochs=3)
