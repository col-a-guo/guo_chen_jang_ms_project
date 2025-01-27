import tokenizers
from tqdm import tqdm
from transformers import BertTokenizerFast
from datasets import load_dataset
import os

# Tokenize the text
bwpt = tokenizers.BertWordPieceTokenizer()

key=""

with open("./secrets.txt", "r") as file:
    key = file.read()
    

filepath = "./BERT_pretrain.txt"

bwpt.train(
    files=[filepath],
    vocab_size=50000,
    min_frequency=3,
    limit_alphabet=1000
)

bwpt.save_model('./')

# Use the load_dataset function to create a dataset
raw_datasets = load_dataset("text", data_files=filepath)


# Split the dataset using train_test_split
raw_datasets = raw_datasets["train"].train_test_split(test_size=0.2) # Split the 'train' split into train and test

# repositor id for saving the tokenizer
tokenizer_id="bottleneckBERT"

# create a python generator to dynamically load the data
def batch_iterator(batch_size=10000):
    for i in tqdm(range(0, len(raw_datasets['train']), batch_size)):
        yield raw_datasets['train'][i : i + batch_size]["text"]

# create a tokenizer from existing one to re-use special tokens
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

bert_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32_000)
bert_tokenizer.save_pretrained("tokenizer")

# you need to be logged in to push the tokenizer
bert_tokenizer.push_to_hub(tokenizer_id)
