from transformers import BertTokenizer, BertForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import random_split

# Load the BusinessBERT tokenizer (or any)
model_name = "pborchet/businessBERT"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Data prep
raw_file_dir = 'BERT_pretrain.txt'  

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=raw_file_dir,
    block_size=128
)

# split
train_size = int(0.8 * len(dataset))
test_size = int(0.1 * len(dataset))
eval_size = len(dataset) - train_size - test_size

train_dataset, test_dataset, eval_dataset = random_split(dataset, [train_size, test_size, eval_size])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Training args
output_dir = '/working/'  

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    push_to_hub=True, 
    hub_model_id="colaguo/bottleneckBERTsmall", 
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

#Train and Save
trainer.train()
trainer.save_model(output_dir)  # Save the fine-tuned model

print(f"Fine-tuned model saved to {output_dir} and pushed to colaguo/bottleneckBERTsmall on Hugging Face Hub")