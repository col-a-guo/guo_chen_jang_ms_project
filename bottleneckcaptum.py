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
    Performs a forward pass of the model and returns classification logits.
    """
    model = MODEL_CACHE[(model_version, "model")]
    if model is None:
        return None
    with torch.no_grad(): # Ensure no gradients are computed during prediction
        output = model(inputs, attention_mask=attention_mask)
    return output.logits


def classification_forward_func(inputs, attention_mask=None, class_ind=0, model_version=None):
    """
    Custom forward function to access a specific class's logit for the model.
    Note: For attributions, we often want the output *before* softmax.
          Captum handles the final layer transformation. Let's return logits.
          If using Integrated Gradients on probabilities, use softmax here.
          For LayerIntegratedGradients on embeddings, logits are usually fine.
          Let's check Captum docs - LIG usually works on intermediate layers,
          and the final output for the target neuron is needed. Using softmax
          output for the target class seems standard.
    """
    pred = predict(inputs, attention_mask=attention_mask, model_version=model_version)
    if pred is None:
        return None
    # Return the probability of the target class
    return torch.softmax(pred, dim=1)[:, class_ind]


def update_global_token_ids(model_version):
    """Updates global token IDs based on the tokenizer."""
    tokenizer = MODEL_CACHE[(model_version, "tokenizer")]
    if tokenizer:
        CURRENT_VALUES[model_version]["ref_token_id"] = tokenizer.pad_token_id
        CURRENT_VALUES[model_version]["sep_token_id"] = tokenizer.sep_token_id
        CURRENT_VALUES[model_version]["cls_token_id"] = tokenizer.cls_token_id
        print(f"Updated token IDs for {model_version}: PAD={tokenizer.pad_token_id}, SEP={tokenizer.sep_token_id}, CLS={tokenizer.cls_token_id}")


def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id, model_version):
    """
    Constructs input and reference token ID pairs for sequence classification.
    """
    tokenizer = MODEL_CACHE[(model_version, "tokenizer")]
    if tokenizer is None:
        print(f"Error: Tokenizer not found for {model_version}")
        return None, None
    if ref_token_id is None or sep_token_id is None or cls_token_id is None:
        print(f"Error: Special token IDs not set for {model_version}")
        return None, None

    # Handle potential empty text input
    if not text.strip():
       text_ids = []
    else:
       text_ids = tokenizer.encode(text, add_special_tokens=False)


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


def visualize(text, ground_truth_class, model_version):
    model = MODEL_CACHE[(model_version, "model")]
    tokenizer = MODEL_CACHE[(model_version, "tokenizer")]
    ref_token_id = CURRENT_VALUES[model_version]["ref_token_id"]
    sep_token_id = CURRENT_VALUES[model_version]["sep_token_id"]
    cls_token_id = CURRENT_VALUES[model_version]["cls_token_id"]

    if tokenizer is None or model is None:
        return f"Model and/or tokenizer not loaded for {model_version}."
    if ref_token_id is None or sep_token_id is None or cls_token_id is None:
         return f"Special token IDs not initialized for {model_version}. Check model loading."

    input_ids, ref_input_ids = construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id, model_version)
    if input_ids is None or ref_input_ids is None:
        return f"Error tokenizing input for {model_version}."
    # Handle case where input becomes too short after tokenization (e.g., empty string)
    if input_ids.shape[1] < 2: # Need at least CLS and SEP
         return f"Input text resulted in too few tokens for {model_version}. Requires at least [CLS] and [SEP]."


    attention_mask = construct_attention_mask(input_ids)

    indices = input_ids.squeeze(0).detach().cpu().tolist() # Use cpu() for safety
    all_tokens = tokenizer.convert_ids_to_tokens(indices)

    # Check if Layer Integrated Gradients instance exists, create if not
    if model_version not in LIG_INSTANCES:
        try:
            # Ensure the embedding layer name is correct for the specific BERT variant if needed
            # Common names: model.bert.embeddings, model.embeddings
            embedding_layer = model.bert.embeddings
            LIG_INSTANCES[model_version] = LayerIntegratedGradients(classification_forward_func, embedding_layer)
            print(f"Initialized LayerIntegratedGradients for {model_version}")
        except AttributeError:
             try: # Fallback for potentially different embedding layer name structures
                 embedding_layer = model.embeddings
                 LIG_INSTANCES[model_version] = LayerIntegratedGradients(classification_forward_func, embedding_layer)
                 print(f"Initialized LayerIntegratedGradients for {model_version} (using model.embeddings)")
             except AttributeError:
                 print(f"Error: Could not find embedding layer for {model_version}. Searched model.bert.embeddings and model.embeddings.")
                 return f"Error initializing LIG for {model_version}: Embedding layer not found."
        except Exception as e:
            print(f"Error initializing LayerIntegratedGradients for {model_version}: {e}")
            return f"Error initializing LIG for {model_version}."


    lig = LIG_INSTANCES[model_version]

    # Get model prediction *before* calculating attributions
    logits = predict(input_ids, attention_mask=attention_mask, model_version=model_version)
    if logits is None:
        return f"Error during prediction for {model_version}."
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item() # Use dim=1 and .item()
    predicted_probability = probabilities[0, predicted_class].item() # Index correctly

    print(f'-- {model_version} --')
    print(f'Text: "{text}"')
    print(f'Tokens: {all_tokens}')
    print(f'Predicted Class: {predicted_class} (Prob: {predicted_probability:.4f})')
    print(f'Ground Truth Class: {ground_truth_class}')
    # print(f'Probabilities: {probabilities.detach().cpu().numpy()}') # Optional: print all probs

    # --- Attribution Calculation ---
    # We attribute w.r.t the *ground truth* class to see why the model made its prediction
    # relative to what it *should* have predicted (or w.r.t predicted class if preferred)
    # Let's attribute w.r.t the *predicted* class to see explanation for the prediction made.
    # If you want to see why it *failed* (if pred != true), attribute w.r.t ground_truth_class.
    attribution_target_class = predicted_class # Or use ground_truth_class

    try:
        attributions, delta = lig.attribute(inputs=input_ids,
                                          baselines=ref_input_ids,
                                          additional_forward_args=(attention_mask, attribution_target_class, model_version),
                                          return_convergence_delta=True,
                                          target=attribution_target_class) # Pass target explicitly if needed by forward_func wrapper, though already in args
    except Exception as e:
        print(f"Error during attribution calculation for {model_version}: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        return f"Error during attribution calculation for {model_version}."

    def summarize_attributions(attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions.cpu().detach() # Move to CPU and detach


    attributions_sum = summarize_attributions(attributions)

    print(f'Convergence Delta ({model_version}): {delta.item()}') # Use .item()

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


def process_input(text_input, ground_truth_input_str):
    """Processes the input text and ground truth for all models."""
    results = {}
    try:
        # Ensure ground truth is an integer
        ground_truth_input = int(ground_truth_input_str)
    except (ValueError, TypeError):
         # Handle invalid number input gracefully for all models
        error_msg = "Invalid Ground Truth: Please enter an integer (e.g., 0 or 1)."
        print(error_msg)
        for model_version in MODEL_VERSIONS:
             results[model_version] = f"<p style='color:red;'>{error_msg}</p>"
        return results # Return dict of error messages

    for model_version in MODEL_VERSIONS:
        model = MODEL_CACHE.get((model_version, "model"))
        tokenizer = MODEL_CACHE.get((model_version, "tokenizer"))
        if not model or not tokenizer:
            results[model_version] = f"<p style='color:orange;'>Model or Tokenizer for {model_version} not loaded. Skipping.</p>"
            print(f"Skipping {model_version} due to missing model/tokenizer.")
            continue

        # Check if special tokens are loaded, otherwise skip
        if CURRENT_VALUES[model_version].get("ref_token_id") is None:
             results[model_version] = f"<p style='color:orange;'>Special token IDs not set for {model_version}. Skipping.</p>"
             print(f"Skipping {model_version} due to missing special token IDs.")
             continue

        print(f"\n--- Processing for {model_version} ---")
        try:
            viz_html = visualize(text_input, ground_truth_input, model_version)
            results[model_version] = viz_html
            CURRENT_VALUES[model_version]["current_text"] = text_input
            CURRENT_VALUES[model_version]["current_ground_truth"] = ground_truth_input
        except Exception as e:
             print(f"Error processing {model_version}: {e}")
             import traceback
             traceback.print_exc()
             results[model_version] = f"<p style='color:red;'>Error during visualization for {model_version}: {e}</p>"

    return results



# --- Load models and set up Gradio Interface ---

# Load all models at startup
print("--- Loading Models ---")
for model_version in MODEL_VERSIONS:
    model, tokenizer = load_model_and_tokenizer(model_version)
    if model and tokenizer:
        update_global_token_ids(model_version)
        # LIG instances are now created lazily inside visualize() if needed
    else:
        print(f"--- Failed to load {model_version} ---")
print("--- Model Loading Complete ---")


with gr.Blocks() as iface:
    gr.Markdown("# BERT Visualization with Multiple Models")
    gr.Markdown("Enter text and the ground truth class (0 or 1). Attributions explain the model's *predicted* class.")

    with gr.Row():
        text_input = gr.Textbox(lines=2, placeholder="Enter text here...", label="Input Text", scale=3)
        ground_truth_input = gr.Number(value=0, label="Ground Truth Class", minimum=0, maximum=1, step=1, scale=1) # Assuming binary classification

    process_button = gr.Button("Visualize Attributions for All Models")

    output_htmls = {}
    with gr.Column(): # Arrange outputs vertically
        for model_version in MODEL_VERSIONS:
            # Add a markdown header for each model's output section
            gr.Markdown(f"### {model_version}")
            output_htmls[model_version] = gr.HTML(label=f"Attribution Visualization - {model_version}") # Label might not be prominently displayed depending on theme/layout

    def update_visualizations_interface(text, ground_truth):
        # This wrapper ensures the results dict is unpacked correctly for Gradio outputs
        results_dict = process_input(text, ground_truth)
        # Return results in the specific order Gradio expects based on the outputs list
        return [results_dict[model_version] for model_version in MODEL_VERSIONS]

    process_button.click(
        update_visualizations_interface, # Use the interface wrapper function
        inputs=[text_input, ground_truth_input],
        # Ensure this list matches the order of MODEL_VERSIONS used to create output_htmls
        outputs=[output_htmls[model_version] for model_version in MODEL_VERSIONS],
    )

# Launch the Gradio interface
iface.launch(debug=True) # Add debug=True for easier troubleshooting during development