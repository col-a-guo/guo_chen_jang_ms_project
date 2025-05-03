
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

# Dictionary to store loaded models and tokenizers
MODEL_CACHE = {}
current_model = None
current_tokenizer = None
current_model_version = None
lig = None  # Initialize LayerIntegratedGradients outside

def load_model_and_tokenizer(model_version):
    """Loads the specified model and tokenizer, or retrieves from cache."""
    global MODEL_CACHE
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
    Performs a forward pass of the current model and returns classification logits.
    """
    if current_model is None:
        return None
    output = current_model(inputs, attention_mask=attention_mask)
    return output.logits


def classification_forward_func(inputs, attention_mask=None, class_ind=0, model_version=None):
    """
    Custom forward function to access a specific class's logit for the current model.
    """
    pred = predict(inputs, attention_mask=attention_mask)
    if pred is None:
        return None
    return torch.softmax(pred, dim=1)[:, class_ind] # Access logit for class_ind


ref_token_id = None
sep_token_id = None
cls_token_id = None

def update_global_token_ids():
    """Updates global token IDs based on the current tokenizer."""
    global ref_token_id, sep_token_id, cls_token_id
    if current_tokenizer:
        ref_token_id = current_tokenizer.pad_token_id
        sep_token_id = current_tokenizer.sep_token_id
        cls_token_id = current_tokenizer.cls_token_id

def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id, model_version):
    """
    Constructs input and reference token ID pairs for sequence classification using the current tokenizer.
    """
    if current_tokenizer is None:
        return None, None
    text_ids = current_tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]

    # construct reference token ids
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device)


def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)


def visualize(text, ground_truth_class=0):
    global current_model, current_tokenizer, ref_token_id, sep_token_id, cls_token_id, lig

    if current_tokenizer is None or current_model is None:
        return "Model and tokenizer not loaded."

    input_ids, ref_input_ids = construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id)
    if input_ids is None:
        return "Error tokenizing input."
    attention_mask = construct_attention_mask(input_ids)

    indices = input_ids.squeeze().detach().tolist()
    all_tokens = current_tokenizer.convert_ids_to_tokens(indices)

    logits = predict(input_ids, attention_mask=attention_mask)
    if logits is None:
        return "Error during prediction."
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities).item()

    print(f'Model: {model_version}, Text: ', text)
    print(f'Model: {model_version}, Predicted Class: ', predicted_class)
    print(f'Model: {model_version}, Probabilities: ', probabilities)

    # Re-initialize LayerIntegratedGradients with the current model
    lig = LayerIntegratedGradients(classification_forward_func, current_model.bert.embeddings)

    attributions, delta = lig.attribute(inputs=input_ids,
                                      baselines=ref_input_ids,
                                      additional_forward_args=(attention_mask, ground_truth_class),
                                      return_convergence_delta=True)


    def summarize_attributions(attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions


    attributions_sum = summarize_attributions(attributions)

    predicted_probability = probabilities[:, predicted_class].item()
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

def switch_model(model_version):
    """Switches the currently active model and tokenizer."""
    global current_model_version, current_model, current_tokenizer, lig
    if model_version != current_model_version:
        print(f"Switching model to: {model_version}")
        new_model, new_tokenizer = load_model_and_tokenizer(model_version)
        if new_model and new_tokenizer:
            current_model = new_model
            current_tokenizer = new_tokenizer
            current_model_version = model_version
            update_global_token_ids()
            # Importantly, reset the LayerIntegratedGradients object
            lig = LayerIntegratedGradients(classification_forward_func, current_model.bert.embeddings)
            return f"Model switched to {model_version}"
        else:
            return f"Failed to load model {model_version}"
    else:
        return f"Already using model {model_version}"

# Define the Gradio interface with buttons
with gr.Blocks() as iface:
    gr.Markdown("# BERT Visualization with Model Switching")

    with gr.Row():
        model_choice = gr.Radio(["bert-uncased", "businessBERT", "bottleneckBERT"],
                                label="Choose Model", value="bert-uncased") # Set initial value
        switch_button = gr.Button("Switch Model")
        switch_status = gr.Textbox(label="Model Status", value="Current model: None")

    text_input = gr.Textbox(lines=2, placeholder="Enter text here...")
    ground_truth_input = gr.Number(value=0, label="Ground Truth Class")
    visualize_button = gr.Button("Visualize Attributions")
    output_html = gr.HTML(label="Attribution Visualization")

    def initial_load(model_version):
        global current_model_version, current_model, current_tokenizer, lig
        current_model, current_tokenizer = load_model_and_tokenizer(model_version)
        current_model_version = model_version
        update_global_token_ids()
        if current_model:
            lig = LayerIntegratedGradients(classification_forward_func, current_model.bert.embeddings)
            return f"Current model: {current_model_version}"
        else:
            return "Failed to load initial model."

    initial_status = initial_load("bert-uncased")
    switch_status.value = initial_status

    switch_button.click(switch_model, inputs=model_choice, outputs=switch_status)
    visualize_button.click(visualize, inputs=[text_input, ground_truth_input], outputs=output_html)
    model_choice.change(switch_model, inputs=model_choice, outputs=switch_status) # Optional: Switch on radio change

# Launch the Gradio interface
iface.launch()
