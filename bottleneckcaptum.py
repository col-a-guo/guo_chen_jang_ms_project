
import numpy as np
import pandas as pd
# import seaborn as sns # Not used in the final code, can be removed if not needed elsewhere
# import matplotlib.pyplot as plt # Not used in the final code, can be removed if not needed elsewhere
from lxml import html as lxml_html
import torch
import torch.nn as nn

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients

import gradio as gr
from bs4 import BeautifulSoup # Import BeautifulSoup

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
        # Adjust path if necessary, assuming they are in subfolders like 'models/bert-uncased_finetune_feb24'
        # Or if using Hugging Face Hub paths like 'colaguo/bert-uncased_finetune_feb24'
        model_name = f"colaguo/{model_version}_finetune_feb24"
        try:
            print(f"Attempting to load: {model_name}")
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
    # Return the probability of the target class
    return torch.softmax(pred, dim=1)[:, class_ind]


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

    # Ensure tensors have at least one dimension if text is empty
    if not input_ids:
        print("Warning: Empty text resulted in empty token sequence.")
        # Handle appropriately, maybe return empty tensors or raise error
        # For now, let's create minimal valid tensors if possible
        input_ids = [cls_token_id, sep_token_id]
        ref_input_ids = [cls_token_id, sep_token_id]


    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device)


def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)


def remove_attribution_label_column(html_content):
    """Uses BeautifulSoup to remove the 'Attribution Label' column from Captum's HTML table."""
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')

    if not table:
        print("Warning: Could not find table in Captum HTML output.")
        return html_content # Return original if no table found

    target_header_text = "Attribution Label"
    target_header_index = -1

    # Find header index and remove the header cell
    thead = table.find('thead')
    header_row = None
    if thead:
        header_row = thead.find('tr')
        if header_row:
            header_cells = header_row.find_all('th')
            for i, th in enumerate(header_cells):
                if th.get_text(strip=True) == target_header_text:
                    target_header_index = i
                    th.decompose() # Remove the header cell
                    break

    # Fallback if no thead or header not found in thead
    if target_header_index == -1:
        first_row = table.find('tr') # Assume first row might contain headers
        if first_row:
            header_cells = first_row.find_all('th')
            if header_cells: # Check if it actually has th elements
                 for i, th in enumerate(header_cells):
                     if th.get_text(strip=True) == target_header_text:
                        target_header_index = i
                        th.decompose()
                        break

    if target_header_index == -1:
        print(f"Warning: '{target_header_text}' header not found. Cannot remove column.")
        # Optionally, inspect the first few td elements of the first row if no th exists
        # This is less robust as structure might vary significantly
        return str(soup) # Return HTML modified so far (or original if no table found)

    # Remove data cells in the corresponding column
    tbody = table.find('tbody')
    if tbody:
        data_rows = tbody.find_all('tr')
        for row in data_rows:
            cells = row.find_all('td')
            if len(cells) > target_header_index:
                cells[target_header_index].decompose()
    # Handle case where data rows might be directly under table (no tbody)
    else:
        data_rows = table.find_all('tr')
        # Skip header row if it was found earlier
        start_index = 1 if header_row else 0
        for row in data_rows[start_index:]:
            cells = row.find_all('td')
            if len(cells) > target_header_index:
                cells[target_header_index].decompose()


    return str(soup) # Return the modified HTML string


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
        return f"Error during prediction for {model_version}."
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item() # Use dim=1 and .item()
    predicted_probability = probabilities[0, predicted_class].item() # Index correctly

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
        return attributions.cpu().detach() # Move to CPU and detach


    attributions_sum = summarize_attributions(attributions)

    predicted_probability = probabilities[:, predicted_class].item()
    predicted_probability = probabilities[:, predicted_class].item()

    vis_data_record = viz.VisualizationDataRecord(
                            word_attributions=attributions_sum, # Use named arg for clarity
                            pred_prob=predicted_probability,
                            pred_class=predicted_class,
                            true_class=ground_truth_class,
                            attr_label=str(ground_truth_class), # This is the label we want to remove
                            attr_score=attributions_sum.sum().item(), # Use .item()
                            raw_input_ids=all_tokens, # Use named arg
                            convergence_score=delta.item()) # Use named arg and .item()

    # Generate the original HTML visualization
    html_obj = viz.visualize_text([vis_data_record])
    original_html = html_obj.data

    # Remove the "Attribution Label" column using BeautifulSoup
    modified_html = remove_attribution_label_column(original_html)

    return modified_html # Return the modified HTML string

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
