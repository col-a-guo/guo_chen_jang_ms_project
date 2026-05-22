import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving images
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import torch

from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig
from captum.attr import LayerIntegratedGradients

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = "dec13_combined.csv"
OUTPUT_DIR     = "heatmap_outputs_mixed"
MAX_SAMPLES    = 60
MAX_PER_CLASS  = 20    # collect exactly this many per stage (34×3 = 102, stops at 100)
MODEL_VERSIONS = ["bert-uncased", "businessBERT", "bottleneckBERT"]
NUM_CLASSES    = 3          # <-- 3-class: stages 0, 1, 2

HF_REPO_MAP = {
    "bottleneckBERT": "colaguo/bottleneckBERT-3",
    "businessBERT":   "colaguo/businessBERT-finetune-3",
    "bert-uncased":   "colaguo/BERT-finetune-3",
}

device = torch.device("cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SEQ_LEN = 512   # BERT hard limit

# ── RGB triangle color mixing ─────────────────────────────────────────────────
# Class 0 → red   (1, 0, 0)
# Class 1 → green (0, 1, 0)
# Class 2 → blue  (0, 0, 1)
CLASS_COLORS = np.array([
    [0.84, 0.15, 0.16],   # red
    [0.17, 0.63, 0.17],   # green
    [0.12, 0.47, 0.71],   # blue
], dtype=np.float32)

def weights_to_rgb(weights):
    """
    weights: (n_tokens, 3) non-negative floats, already normalised per token.
    Returns (n_tokens, 3) RGB array in [0, 1].
    Blends the three class colours by attribution weight.
    Tokens with near-zero total attribution get a neutral grey.
    """
    total  = weights.sum(axis=1, keepdims=True)
    safe   = np.where(total > 1e-6, total, 1.0)
    w_norm = weights / safe                        # (n, 3) — rows sum to 1
    rgb    = w_norm @ CLASS_COLORS                 # (n, 3)
    # fade toward grey for low-attribution tokens
    grey   = np.full_like(rgb, 0.88)
    alpha  = np.clip(total / (total.max() + 1e-6), 0, 1)
    rgb    = alpha * rgb + (1 - alpha) * grey
    return np.clip(rgb, 0, 1)

# ── Caches ────────────────────────────────────────────────────────────────────
MODEL_CACHE = {}
LIG_CACHE   = {}
TOKEN_IDS   = {}

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(version):
    if (version, "model") in MODEL_CACHE:
        return
    name = HF_REPO_MAP[version]
    print(f"Loading {name} ...")

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config    = BertConfig.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)

    try:
        ckpt_path  = hf_hub_download(repo_id=name, filename="model.safetensors")
        state_dict = load_file(ckpt_path, device=str(device))
    except Exception:
        print(f"  model.safetensors not found, falling back to pytorch_model.bin")
        ckpt_path  = hf_hub_download(repo_id=name, filename="pytorch_model.bin")
        state_dict = torch.load(ckpt_path, map_location=device)

    # Some checkpoints were trained with a custom vocab size — read it directly
    # from the saved weights so the model architecture matches exactly.
    embed_key = "bert.embeddings.word_embeddings.weight"
    if embed_key in state_dict:
        ckpt_vocab_size = state_dict[embed_key].shape[0]
        if ckpt_vocab_size != config.vocab_size:
            print(f"  Patching vocab size: config={config.vocab_size} -> checkpoint={ckpt_vocab_size}")
            config.vocab_size = ckpt_vocab_size

    # Override num_labels to match 3-class setup
    config.num_labels = NUM_CLASSES

    model = BertForSequenceClassification(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Weights loaded  (missing={len(missing)}, unexpected={len(unexpected)})")

    model.to(device).eval()
    model.zero_grad()
    MODEL_CACHE[(version, "model")]     = model
    MODEL_CACHE[(version, "tokenizer")] = tokenizer
    TOKEN_IDS[version] = {
        "ref": tokenizer.pad_token_id,
        "sep": tokenizer.sep_token_id,
        "cls": tokenizer.cls_token_id,
    }
    LIG_CACHE[version] = LayerIntegratedGradients(
        lambda inp, attn, cls_idx, ver: torch.softmax(
            MODEL_CACHE[(ver, "model")](inp, attention_mask=attn).logits, dim=1
        )[:, cls_idx],
        model.bert.embeddings,
    )
    print(f"  OK {version}")

# ── Tokenisation helpers ───────────────────────────────────────────────────────
def build_input_ref(text, version):
    tok = MODEL_CACHE[(version, "tokenizer")]
    if version == "businessBERT":
        text = text.lower()
    ids = tok.encode(text, add_special_tokens=False)
    ids = ids[: MAX_SEQ_LEN - 2]          # leave room for [CLS] and [SEP]
    t   = TOKEN_IDS[version]
    inp = torch.tensor([[t["cls"]] + ids + [t["sep"]]], device=device)
    ref = torch.tensor([[t["cls"]] + [t["ref"]] * len(ids) + [t["sep"]]], device=device)
    return inp, ref

def get_tokens(input_ids, version):
    return MODEL_CACHE[(version, "tokenizer")].convert_ids_to_tokens(
        input_ids.squeeze().tolist()
    )

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(input_ids, version):
    attn   = torch.ones_like(input_ids)
    logits = MODEL_CACHE[(version, "model")](input_ids, attention_mask=attn).logits
    probs  = torch.softmax(logits, dim=1)
    return torch.argmax(probs).item(), probs

# ── Attribution ───────────────────────────────────────────────────────────────
def get_attributions(text, version):
    """
    Compute integrated gradients for all 3 classes independently.
    Returns:
      tokens      : list of str
      rgb_weights : (n_tokens, 3) array of non-negative attribution magnitudes,
                    one column per class (0=red, 1=green, 2=blue).
    """
    inp, ref = build_input_ref(text, version)
    attn     = torch.ones_like(inp)
    tokens   = get_tokens(inp, version)
    lig      = LIG_CACHE[version]

    per_class = []
    for cls_idx in range(NUM_CLASSES):
        attrs, _ = lig.attribute(
            inputs=inp,
            baselines=ref,
            additional_forward_args=(attn, cls_idx, version),
            return_convergence_delta=True,
        )
        attrs_sum = attrs.sum(dim=-1).squeeze(0)          # (seq_len,)
        # keep only positive contributions for this class
        positive  = torch.clamp(attrs_sum, min=0).detach().numpy()
        per_class.append(positive)

    rgb_weights = np.stack(per_class, axis=1)             # (seq_len, 3)
    return tokens, rgb_weights

# ── Class label helper ────────────────────────────────────────────────────────
STAGE_LABELS = {0: "Stage 0", 1: "Stage 1", 2: "Stage 2"}

def class_label(cls_idx):
    return STAGE_LABELS.get(int(cls_idx), str(cls_idx))

# ── Paragraph-style panel renderer ───────────────────────────────────────────
def render_paragraph_panel(fig, ax, tokens, rgb_weights, version, pred_class, true_class):
    """
    Renders tokens as a flowing paragraph of highlighted text.
    Each token's background is a blend of red / green / blue proportional
    to its positive attribution toward Stage 0 / 1 / 2 respectively.
    Pure grey = no strong attribution to any class.
    """
    # Compute per-token RGB colours
    colors_rgb = weights_to_rgb(rgb_weights)   # (n_tokens, 3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("#f7f7f7")

    # Title
    title = (
        f"{version}   |   "
        f"Predicted: {class_label(pred_class)}   |   "
        f"True label: {class_label(true_class)}"
    )
    ax.text(0.01, 0.975, title,
            transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            va="top", ha="left", color="#111111",
            fontfamily="monospace")

    # Legend patches (top-right)
    for i, (label_str, color) in enumerate(
        zip(["Stage 0", "Stage 1", "Stage 2"], CLASS_COLORS)
    ):
        bx = 0.75 + i * 0.08
        rect = FancyBboxPatch((bx, 0.955), 0.06, 0.03,
                              boxstyle="round,pad=0.001", linewidth=0,
                              facecolor=color, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(bx + 0.03, 0.97, label_str,
                transform=ax.transAxes, fontsize=7,
                va="center", ha="center", color="white", fontweight="bold")

    # Layout constants
    FONT_SIZE      = 9
    CHARS_PER_LINE = 100
    LINE_H         = 0.055
    X_START        = 0.01
    Y_START        = 0.91
    CHAR_W         = (1.0 - X_START * 2) / CHARS_PER_LINE
    PAD_Y          = 0.006

    # Filter out special / padding tokens
    filtered = [
        (tok, col)
        for tok, col in zip(tokens, colors_rgb)
        if tok not in ("[PAD]", "[CLS]", "[SEP]", "")
    ]

    # Wrap into lines
    lines        = []
    current_line = []
    current_len  = 0
    for tok, col in filtered:
        tok_len = len(tok) + 1
        if current_len + tok_len > CHARS_PER_LINE and current_line:
            lines.append(current_line)
            current_line = [(tok, col)]
            current_len  = tok_len
        else:
            current_line.append((tok, col))
            current_len += tok_len
    if current_line:
        lines.append(current_line)

    max_lines = int((Y_START - 0.08) / LINE_H)
    truncated  = len(lines) > max_lines
    lines      = lines[:max_lines]

    for line_idx, line_tokens in enumerate(lines):
        y = Y_START - line_idx * LINE_H
        x = X_START

        for tok, color in line_tokens:
            tok_w = len(tok) * CHAR_W
            box_w = tok_w + CHAR_W * 0.5

            rect = FancyBboxPatch(
                (x, y - LINE_H + PAD_Y),
                box_w, LINE_H - PAD_Y,
                boxstyle="round,pad=0.001",
                linewidth=0,
                facecolor=color,
                alpha=0.85,
                transform=ax.transAxes,
                clip_on=True,
            )
            ax.add_patch(rect)

            ax.text(x + CHAR_W * 0.25,
                    y - LINE_H / 2,
                    tok,
                    transform=ax.transAxes,
                    fontsize=FONT_SIZE,
                    fontfamily="monospace",
                    va="center", ha="left",
                    color="black",
                    clip_on=True)

            x += tok_w + CHAR_W

    if truncated:
        trunc_y = Y_START - len(lines) * LINE_H - 0.01
        ax.text(X_START, trunc_y, "... (truncated)",
                transform=ax.transAxes, fontsize=7, color="#888888")


# ── Stitch all 3 panels into one PNG ─────────────────────────────────────────
def save_combined(all_model_data, sample_idx):
    """
    all_model_data: list of (tokens, rgb_weights, version, pred_class, true_class)
    Renders one panel per model, stitched vertically, saved as a single PNG.
    """
    n = len(all_model_data)
    fig, axes = plt.subplots(n, 1,
                             figsize=(16, 5 * n),
                             facecolor="white")
    if n == 1:
        axes = [axes]

    for ax, (tokens, attrs, version, pred_cls, true_cls) in zip(axes, all_model_data):
        render_paragraph_panel(fig, ax, tokens, attrs, version, pred_cls, true_cls)

    fig.suptitle(f"Sample {sample_idx}", fontsize=13, fontweight="bold", y=1.002)
    plt.tight_layout()

    fname = os.path.join(OUTPUT_DIR, f"sample_{sample_idx:04d}.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fname


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    # Validate labels are 0/1/2
    unique_labels = df["label"].unique()
    print(f"Unique labels in dataset: {sorted(unique_labels)}")
    assert set(unique_labels).issubset({0, 1, 2, 0.0, 1.0, 2.0}), \
        f"Expected labels {{0,1,2}}, got {unique_labels}"

    # Pre-stratify: take up to N rows per class so the search is balanced.
    # Stage 2 gets more rows because bottleneckBERT-wins are rarer there.
    CLASS_BUDGET = {0: 50, 1: 50, 2: 50}
    df = (
        df.groupby("label", group_keys=False)
          .apply(lambda g: g.sample(n=min(len(g), CLASS_BUDGET[int(g.name)]),
                                    random_state=42))
          .reset_index(drop=True)
          .sample(frac=1, random_state=42)   # shuffle the interleaved result
          .reset_index(drop=True)
    )
    print(f"Stratified pool: {dict(df['label'].value_counts().sort_index())}")

    for v in MODEL_VERSIONS:
        load_model(v)
 
 

    found   = 0
    row_idx = 0

    while found < MAX_SAMPLES and row_idx < len(df):
        text  = df.iloc[row_idx]["paragraph"]
        label = int(df.iloc[row_idx]["label"])

        preds = {}
        for v in MODEL_VERSIONS:
            inp, _      = build_input_ref(text, v)
            preds[v], _ = predict(inp, v)

        bottleneck_ok = preds.get("bottleneckBERT") == label
        others_wrong  = all(preds[v] != label for v in MODEL_VERSIONS if v != "bottleneckBERT")

        if bottleneck_ok and others_wrong:
            found += 1
            print(f"\n-- Sample {found} (df row {row_idx})  label={class_label(label)} --")

            panel_data = []
            for v in MODEL_VERSIONS:
                inp, _          = build_input_ref(text, v)
                pred_cls, _     = predict(inp, v)
                tokens, rgb_w   = get_attributions(text, v)
                panel_data.append((tokens, rgb_w, v, pred_cls, label))
                print(f"  {v:20s}  pred={class_label(pred_cls)}")

            fpath = save_combined(panel_data, found)
            print(f"  -> {fpath}")

        row_idx += 1

    print(f"\nDone. Saved {found} samples to '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()