import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig

from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------
#  LOAD YOUR FINE-TUNED MODEL AND TOKENIZER HERE
# ----------------------------------------------------------------------------------
# Replace <PATH-TO-SAVED-MODEL> with the actual path to your saved model
model_path = 'path/to/your/fine/tuned/model'

# Load model
model = BertForQuestionAnswering.from_pretrained(model_path)
model.to(device)
model.eval()
model.zero_grad()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
# ----------------------------------------------------------------------------------

def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    """
    Performs a forward pass of the model and returns start and end logits.

    Args:
        inputs (torch.Tensor): Input tensor of token IDs.
        token_type_ids (torch.Tensor, optional): Token type IDs tensor. Defaults to None.
        position_ids (torch.Tensor, optional): Position IDs tensor. Defaults to None.
        attention_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

    Returns:
        tuple(torch.Tensor, torch.Tensor): Start and end logits.
    """
    output = model(inputs, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask, )
    return output.start_logits, output.end_logits


def squad_pos_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
    """
    Custom forward function to access start or end logits based on position.

    Args:
        inputs (torch.Tensor): Input tensor of token IDs.
        token_type_ids (torch.Tensor, optional): Token type IDs tensor. Defaults to None.
        position_ids (torch.Tensor, optional): Position IDs tensor. Defaults to None.
        attention_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
        position (int, optional): 0 for start logits, 1 for end logits. Defaults to 0.

    Returns:
        torch.Tensor: Max logit value for the specified position.
    """
    pred = predict(inputs,
                   token_type_ids=token_type_ids,
                   position_ids=position_ids,
                   attention_mask=attention_mask)
    pred = pred[position]
    return pred.max(1).values


ref_token_id = tokenizer.pad_token_id  # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id  # Separator token
cls_token_id = tokenizer.cls_token_id  # CLS token


def construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id):
    """
    Constructs input and reference token ID pairs.

    Args:
        question (str): Question string.
        text (str): Text string.
        ref_token_id (int): Reference token ID.
        sep_token_id (int): Separator token ID.
        cls_token_id (int): CLS token ID.

    Returns:
        tuple(torch.Tensor, torch.Tensor, int): Input token IDs, reference token IDs, and question length.
    """
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + question_ids + [sep_token_id] + text_ids + [sep_token_id]

    # construct reference token ids
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] + \
        [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(question_ids)


def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    """
    Constructs token type ID pairs.

    Args:
        input_ids (torch.Tensor): Input token IDs.
        sep_ind (int, optional): Index of the separator token. Defaults to 0.

    Returns:
        tuple(torch.Tensor, torch.Tensor): Token type IDs and reference token type IDs.
    """
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)  # * -1
    return token_type_ids, ref_token_type_ids


def construct_input_ref_pos_id_pair(input_ids):
    """
    Constructs position ID pairs.

    Args:
        input_ids (torch.Tensor): Input token IDs.

    Returns:
        tuple(torch.Tensor, torch.Tensor): Position IDs and reference position IDs.
    """
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids


def construct_attention_mask(input_ids):
    """
    Constructs attention mask.

    Args:
        input_ids (torch.Tensor): Input token IDs.

    Returns:
        torch.Tensor: Attention mask.
    """
    return torch.ones_like(input_ids)


def construct_whole_bert_embeddings(input_ids, ref_input_ids,
                                    token_type_ids=None, ref_token_type_ids=None,
                                    position_ids=None, ref_position_ids=None):
    """
    Constructs BERT embeddings for input and reference token IDs.

    Args:
        input_ids (torch.Tensor): Input token IDs.
        ref_input_ids (torch.Tensor): Reference token IDs.
        token_type_ids (torch.Tensor, optional): Token type IDs. Defaults to None.
        ref_token_type_ids (torch.Tensor, optional): Reference token type IDs. Defaults to None.
        position_ids (torch.Tensor, optional): Position IDs. Defaults to None.
        ref_position_ids (torch.Tensor, optional): Reference position IDs. Defaults to None.

    Returns:
        tuple(torch.Tensor, torch.Tensor): Input embeddings and reference embeddings.
    """
    input_embeddings = model.bert.embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    ref_input_embeddings = model.bert.embeddings(ref_input_ids, token_type_ids=ref_token_type_ids, position_ids=ref_position_ids)

    return input_embeddings, ref_input_embeddings


# ----------------------------------------------------------------------------------
#  YOUR QUESTION AND TEXT HERE
# ----------------------------------------------------------------------------------
question = "What is important to us?"
text = "It is important to us to include, empower and support humans of all kinds."
# ----------------------------------------------------------------------------------

input_ids, ref_input_ids, sep_id = construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id)
token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
attention_mask = construct_attention_mask(input_ids)

indices = input_ids[0].detach().tolist()
all_tokens = tokenizer.convert_ids_to_tokens(indices)

# ----------------------------------------------------------------------------------
#  YOUR GROUND TRUTH ANSWER HERE
# ----------------------------------------------------------------------------------
ground_truth = 'to include, empower and support humans of all kinds'

ground_truth_tokens = tokenizer.encode(ground_truth, add_special_tokens=False)
try:
    ground_truth_end_ind = indices.index(ground_truth_tokens[-1])
    ground_truth_start_ind = ground_truth_end_ind - len(ground_truth_tokens) + 1
except ValueError:
    print("Warning: Ground truth answer not found in input text.")
    ground_truth_start_ind = 0
    ground_truth_end_ind = 0
# ----------------------------------------------------------------------------------

start_scores, end_scores = predict(input_ids,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   attention_mask=attention_mask)


print('Question: ', question)
print('Predicted Answer: ', ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores)+1]))


lig = LayerIntegratedGradients(squad_pos_forward_func, model.bert.embeddings)

attributions_start, delta_start = lig.attribute(inputs=input_ids,
                                  baselines=ref_input_ids,
                                  additional_forward_args=(token_type_ids, position_ids, attention_mask, 0),
                                  return_convergence_delta=True)
attributions_end, delta_end = lig.attribute(inputs=input_ids, baselines=ref_input_ids,
                                additional_forward_args=(token_type_ids, position_ids, attention_mask, 1),
                                return_convergence_delta=True)


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


attributions_start_sum = summarize_attributions(attributions_start)
attributions_end_sum = summarize_attributions(attributions_end)


start_position_vis = viz.VisualizationDataRecord(
                        attributions_start_sum,
                        torch.max(torch.softmax(start_scores[0], dim=0)),
                        torch.argmax(start_scores),
                        torch.argmax(start_scores),
                        str(ground_truth_start_ind),
                        attributions_start_sum.sum(),
                        all_tokens,
                        delta_start)

end_position_vis = viz.VisualizationDataRecord(
                        attributions_end_sum,
                        torch.max(torch.softmax(end_scores[0], dim=0)),
                        torch.argmax(end_scores),
                        torch.argmax(end_scores),
                        str(ground_truth_end_ind),
                        attributions_end_sum.sum(),
                        all_tokens,
                        delta_end)

print('\033[1m', 'Visualizations For Start Position', '\033[0m')
viz.visualize_text([start_position_vis])

print('\033[1m', 'Visualizations For End Position', '\033[0m')
viz.visualize_text([end_position_vis])