# Load the tokenizer
from transformers import BertTokenizer, LineByLineTextDataset
from torch.utils.data import random_split

vocab_file_dir = 'vocab.txt'
raw_file_dir = 'BERT_pretrain.txt'

tokenizer = BertTokenizer.from_pretrained('pborchert/BusinessBERT')

# sentence = 'Collin is working on business bottlenecks'
# encoded_input = tokenizer.tokenize(sentence)
# print(encoded_input)

# Load the entire dataset
dataset = LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = raw_file_dir,
    block_size = 128  # maximum sequence length
)

print('Total No. of lines: ', len(dataset)) # Total number of lines in your dataset


# --- Splitting the dataset ---
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = int(0.1 * len(dataset))    # 10% for testing
eval_size = len(dataset) - train_size - test_size # remaining 10% for evaluation


train_dataset, test_dataset, eval_dataset = random_split(dataset, [train_size, test_size, eval_size])

print('Training dataset size: ', len(train_dataset))
print('Test dataset size: ', len(test_dataset))
print('Evaluation dataset size: ', len(eval_dataset))
# --- End of dataset splitting ---

from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling

config = BertConfig(
    vocab_size=50000,
    hidden_size=768, 
    num_hidden_layers=6, 
    num_attention_heads=12,
    max_position_embeddings=512
)
 
model = BertForMaskedLM(config)
print('No of parameters: ', model.num_parameters())


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, IntervalStrategy

training_args = TrainingArguments(
    output_dir='/working/',
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=32,
    evaluation_strategy = IntervalStrategy.STEPS,
    eval_steps = 50, # Evaluation and Save happens every 50 steps
    save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
    per_device_eval_batch_size=32,
    save_steps=10_000,
    
    weight_decay=0.01,
    push_to_hub=True,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss', # Specify the metric for early stopping.  Important!
    greater_is_better=False, # False for loss, True for accuracy/F1 etc.  Important!
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset, # Use the training dataset
    eval_dataset=eval_dataset,   # Add the evaluation dataset
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0001)] # added early stopping
)


trainer.train()
trainer.save_model('/bottleneckBERT/')