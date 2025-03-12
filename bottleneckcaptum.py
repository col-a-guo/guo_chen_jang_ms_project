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

model_path = 'colaguo/my-awesome-model'  # Corrected model path

# Load model
model = BertForSequenceClassification.from_pretrained(model_path, ignore_mismatched_sizes=True)
model.to(device)
model.eval()
model.zero_grad()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path, ignore_mismatched_sizes=True)

def predict(inputs, attention_mask=None):
    """
    Performs a forward pass of the model and returns classification logits.

    Args:
        inputs (torch.Tensor): Input tensor of token IDs.
        attention_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

    Returns:
        torch.Tensor: Classification logits.
    """
    output = model(inputs, attention_mask=attention_mask)
    return output.logits


def classification_forward_func(inputs, attention_mask=None, class_ind=0):
    """
    Custom forward function to access a specific class's logit.

    Args:
        inputs (torch.Tensor): Input tensor of token IDs.
        attention_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
        class_ind (int, optional): Index of the class to consider. Defaults to 0.

    Returns:
        torch.Tensor: Logit value for the specified class.
    """
    pred = predict(inputs, attention_mask=attention_mask)
    return torch.softmax(pred, dim=1)[:, class_ind] # Access logit for class_ind


ref_token_id = tokenizer.pad_token_id  # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id  # Separator token
cls_token_id = tokenizer.cls_token_id  # CLS token


def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id):
    """
    Constructs input and reference token ID pairs for sequence classification.  No question needed.

    Args:
        text (str): Text string.
        ref_token_id (int): Reference token ID.
        sep_token_id (int): Separator token ID.
        cls_token_id (int): CLS token ID.

    Returns:
        tuple(torch.Tensor, torch.Tensor): Input token IDs and reference token IDs.
    """
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]

    # construct reference token ids
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device)


def construct_attention_mask(input_ids):
    """
    Constructs attention mask.

    Args:
        input_ids (torch.Tensor): Input token IDs.

    Returns:
        torch.Tensor: Attention mask.
    """
    return torch.ones_like(input_ids)


def construct_whole_bert_embeddings(input_ids, ref_input_ids):
    """
    Constructs BERT embeddings for input and reference token IDs.

    Args:
        input_ids (torch.Tensor): Input token IDs.
        ref_input_ids (torch.Tensor): Reference token IDs.


    Returns:
        tuple(torch.Tensor, torch.Tensor): Input embeddings and reference embeddings.
    """
    input_embeddings = model.bert.embeddings(input_ids)
    ref_input_embeddings = model.bert.embeddings(ref_input_ids)

    return input_embeddings, ref_input_embeddings

def visualize(text, ground_truth_class=0):  # added default value to ground_truth_class
    """
    Generates and returns the visualization as HTML.

    Args:
        text (str): Input text to analyze.
        ground_truth_class (int): The ground truth class.

    Returns:
        str: HTML representation of the visualization.
    """

    input_ids, ref_input_ids = construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id)
    attention_mask = construct_attention_mask(input_ids)

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)

    logits = predict(input_ids, attention_mask=attention_mask)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities).item()

    print('Text: ', text)
    print('Predicted Class: ', predicted_class)
    print('Probabilities: ', probabilities)

    lig = LayerIntegratedGradients(classification_forward_func, model.bert.embeddings) # Remove the need for additional args related to question-answering

    attributions, delta = lig.attribute(inputs=input_ids,
                                      baselines=ref_input_ids,
                                      additional_forward_args=(attention_mask, ground_truth_class), # specify class index
                                      return_convergence_delta=True)


    def summarize_attributions(attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions


    attributions_sum = summarize_attributions(attributions)

    predicted_probability = probabilities[0, predicted_class].item()  # Probability of the predicted class

    vis_data_record = viz.VisualizationDataRecord(
                            attributions_sum,
                            predicted_probability,
                            predicted_class,
                            ground_truth_class,  # Show ground truth as well
                            str(ground_truth_class),
                            attributions_sum.sum(),
                            all_tokens,
                            delta)

    # Capture the HTML output instead of directly printing
    html = viz.visualize_text([vis_data_record])[0]  # Get the HTML object

    return html.data  # Return the .data attribute
# Define the Gradio interface
iface = gr.Interface(
    fn=visualize,
    inputs=[gr.Textbox(lines=2, placeholder="Enter text here..."),
            gr.Number(value=0, label="Ground Truth Class")],
    outputs="html",
    title="BERT Visualization",
    description="Enter a sentence and visualize the attributions using Layer Integrated Gradients."
)

# Launch the Gradio interface
iface.launch()