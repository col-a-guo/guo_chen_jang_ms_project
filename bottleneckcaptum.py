import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients

import gradio as gr

device = torch.device("cpu")

# List of models to load
MODEL_VERSIONS = ["bert-uncased", "businessBERT", "bottleneckBERT"]

# Dictionary to store loaded models and tokenizers
MODEL_CACHE = {}

# Dictionary to store LayerIntegratedGradients instances
LIG_INSTANCES = {}

# Dictionary to store current values
CURRENT_VALUES = {version: {"model": None, "tokenizer": None, "ref_token_id": None, "sep_token_id": None, "cls_token_id": None, "current_text": "", "current_ground_truth": 0} for version in MODEL_VERSIONS}



def load_model_and_tokenizer(model_version):
    """Loads the specified model and tokenizer, or retrieves from cache."""
    if (model_version, "model") not in MODEL_CACHE:
        model_name = f"colaguo/{model_version}_finetune_feb24"
        try:
            config = BertConfig.from_pretrained(model_name)
            model = BertForSequenceClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
            tokenizer = BertTokenizer.from_pretrained(model_name, ignore_mismatched_sizes=True)
            model.to(device)
            model.eval()
            model.zero_grad()
            MODEL_CACHE[(model_version, "model")] = model
            MODEL_CACHE[(model_version, "tokenizer")] = tokenizer
            print(f"Loaded model and tokenizer for: {model_version}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None, None
    return MODEL_CACHE[(model_version, "model")], MODEL_CACHE[(model_version, "tokenizer")]

def predict(inputs, attention_mask=None, model_version=None):
    """
    Performs a forward pass of the model and returns classification logits.
    """
    model = MODEL_CACHE[(model_version, "model")]
    if model is None:
        return None
    output = model(inputs, attention_mask=attention_mask)
    return output.logits


def classification_forward_func(inputs, attention_mask=None, class_ind=0, model_version=None):
    """
    Custom forward function to access a specific class's logit for the model.
    """
    pred = predict(inputs, attention_mask=attention_mask, model_version=model_version)
    if pred is None:
        return None
    return torch.softmax(pred, dim=1)[:, class_ind] # Access logit for class_ind


def update_global_token_ids(model_version):
    """Updates global token IDs based on the tokenizer."""
    tokenizer = MODEL_CACHE[(model_version, "tokenizer")]
    if tokenizer:
        CURRENT_VALUES[model_version]["ref_token_id"] = tokenizer.pad_token_id
        CURRENT_VALUES[model_version]["sep_token_id"] = tokenizer.sep_token_id
        CURRENT_VALUES[model_version]["cls_token_id"] = tokenizer.cls_token_id

def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id, model_version):
    """
    Constructs input and reference token ID pairs for sequence classification.
    """
    tokenizer = MODEL_CACHE[(model_version, "tokenizer")]
    if tokenizer is None:
        return None, None
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]

    # construct reference token ids
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device)


def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)


def visualize(text, ground_truth_class, model_version):
    model = MODEL_CACHE[(model_version, "model")]
    tokenizer = MODEL_CACHE[(model_version, "tokenizer")]
    ref_token_id = CURRENT_VALUES[model_version]["ref_token_id"]
    sep_token_id = CURRENT_VALUES[model_version]["sep_token_id"]
    cls_token_id = CURRENT_VALUES[model_version]["cls_token_id"]

    if tokenizer is None or model is None:
        return "Model and tokenizer not loaded."

    input_ids, ref_input_ids = construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id, model_version)
    if input_ids is None:
        return "Error tokenizing input."
    attention_mask = construct_attention_mask(input_ids)

    indices = input_ids.squeeze().detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)

    logits = predict(input_ids, attention_mask=attention_mask, model_version=model_version)
    if logits is None:
        return "Error during prediction."
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities).item()

    print(f'Model: {model_version}, Text: ', text)
    print(f'Model: {model_version}, Predicted Class: ', predicted_class)
    print(f'Model: {model_version}, Probabilities: ', probabilities)

    # Re-initialize LayerIntegratedGradients with the model
    if model_version not in LIG_INSTANCES:
        LIG_INSTANCES[model_version] = LayerIntegratedGradients(classification_forward_func, model.bert.embeddings)

    lig = LIG_INSTANCES[model_version]

    attributions, delta = lig.attribute(inputs=input_ids,
                                      baselines=ref_input_ids,
                                      additional_forward_args=(attention_mask, ground_truth_class, model_version),
                                      return_convergence_delta=True)


    def summarize_attributions(attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions


    attributions_sum = summarize_attributions(attributions)

    predicted_probability = probabilities[:, predicted_class].item()

    vis_data_record = viz.VisualizationDataRecord(
                            attributions_sum,
                            predicted_probability,
                            predicted_class,
                            ground_truth_class,  # Show ground truth as well
                            str(ground_truth_class),
                            attributions_sum.sum(),
                            all_tokens,
                            delta)

    html = viz.visualize_text([vis_data_record])
    return html.data


def process_input(text_input, ground_truth_input):
    """Processes the input text and ground truth for all models."""
    results = {}
    for model_version in MODEL_VERSIONS:
        model = MODEL_CACHE.get((model_version, "model"))
        tokenizer = MODEL_CACHE.get((model_version, "tokenizer"))
        if not model or not tokenizer:
            results[model_version] = "Model not loaded."
            continue
        results[model_version] = visualize(text_input, int(ground_truth_input), model_version)
        CURRENT_VALUES[model_version]["current_text"] = text_input
        CURRENT_VALUES[model_version]["current_ground_truth"] = ground_truth_input
    return results



# Load all models at startup
for model_version in MODEL_VERSIONS:
    model, tokenizer = load_model_and_tokenizer(model_version)
    if model and tokenizer:
        update_global_token_ids(model_version)
        LIG_INSTANCES[model_version] = LayerIntegratedGradients(classification_forward_func, model.bert.embeddings)
    else:
        print(f"Failed to load {model_version}")


with gr.Blocks() as iface:
    gr.Markdown("# BERT Visualization with Multiple Models")

    text_input = gr.Textbox(lines=2, placeholder="Enter text here...", label="Input Text")
    ground_truth_input = gr.Number(value=0, label="Ground Truth Class")
    process_button = gr.Button("Visualize Attributions for All Models")

    output_htmls = {}
    for model_version in MODEL_VERSIONS:
        output_htmls[model_version] = gr.HTML(label=f"Attribution Visualization - {model_version}")

    def update_visualizations(text_input, ground_truth_input):
        results = process_input(text_input, ground_truth_input)
        return [results[model_version] for model_version in MODEL_VERSIONS]


    process_button.click(
        update_visualizations,
        inputs=[text_input, ground_truth_input],
        outputs=[output_htmls[model_version] for model_version in MODEL_VERSIONS],
    )
# Launch the Gradio interface
iface.launch()