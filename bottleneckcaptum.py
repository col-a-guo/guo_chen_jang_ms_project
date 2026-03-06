import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving images
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch

import torch

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from captum.attr import LayerIntegratedGradients

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = "feb_20_combined.csv"
OUTPUT_DIR     = "heatmap_outputs"
MAX_SAMPLES    = 100
MODEL_VERSIONS = ["bert-uncased", "businessBERT", "bottleneckBERT"]

device = torch.device("cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Caches ────────────────────────────────────────────────────────────────────
MODEL_CACHE = {}
LIG_CACHE   = {}
TOKEN_IDS   = {}

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(version):
    if (version, "model") in MODEL_CACHE:
        return
    name = f"colaguo/{version}_finetune_feb24"
    print(f"Loading {name} ...")

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config    = BertConfig.from_pretrained(name)
    tokenizer = BertTokenizer.from_pretrained(name)

    ckpt_path  = hf_hub_download(repo_id=name, filename="model.safetensors")
    state_dict = load_file(ckpt_path, device=str(device))

    # Some checkpoints were trained with a custom vocab size — read it directly
    # from the saved weights so the model architecture matches exactly.
    embed_key = "bert.embeddings.word_embeddings.weight"
    if embed_key in state_dict:
        ckpt_vocab_size = state_dict[embed_key].shape[0]
        if ckpt_vocab_size != config.vocab_size:
            print(f"  Patching vocab size: config={config.vocab_size} -> checkpoint={ckpt_vocab_size}")
            config.vocab_size = ckpt_vocab_size

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
    ids = tok.encode(text, add_special_tokens=False)
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
def get_attributions(text, target_class, version):
    inp, ref = build_input_ref(text, version)
    attn     = torch.ones_like(inp)
    tokens   = get_tokens(inp, version)

    lig = LIG_CACHE[version]
    attrs, _ = lig.attribute(
        inputs=inp,
        baselines=ref,
        additional_forward_args=(attn, target_class, version),
        return_convergence_delta=True,
    )
    attrs_sum  = attrs.sum(dim=-1).squeeze(0)
    norm       = torch.norm(attrs_sum)
    attrs_norm = (attrs_sum / norm) if norm > 1e-6 else torch.zeros_like(attrs_sum)
    return tokens, attrs_norm.detach().numpy()

# ── Paragraph-style panel renderer ───────────────────────────────────────────
def render_paragraph_panel(fig, ax, tokens, attributions, version, pred_class, true_class):
    """
    Renders tokens as a flowing paragraph of highlighted text.
    Each token gets a coloured background (red = towards predicted, blue = away)
    and the text flows naturally left-to-right, wrapping like a real paragraph.
    """
    cmap = plt.cm.RdBu_r
    vmax = max(np.abs(attributions).max(), 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("#f7f7f7")

    # Title
    title = f"{version}   |   Predicted: {pred_class}   |   True label: {true_class}"
    ax.text(0.01, 0.975, title,
            transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            va="top", ha="left", color="#111111",
            fontfamily="monospace")

    # Layout constants — monospace so every char is the same width
    FONT_SIZE      = 9
    CHARS_PER_LINE = 100        # wrap after this many characters
    LINE_H         = 0.055      # axes-fraction height per text line
    X_START        = 0.01
    Y_START        = 0.91       # start just below the title
    CHAR_W         = (1.0 - X_START * 2) / CHARS_PER_LINE
    PAD_Y          = 0.006      # vertical inner padding for highlight boxes

    # Filter out special / padding tokens
    filtered = [(tok, atr) for tok, atr in zip(tokens, attributions)
                if tok not in ("[PAD]", "[CLS]", "[SEP]", "")]

    # Pre-split tokens into lines by cumulative character count
    lines       = []
    current_line = []
    current_len  = 0
    for tok, atr in filtered:
        tok_len = len(tok) + 1   # +1 for the trailing space
        if current_len + tok_len > CHARS_PER_LINE and current_line:
            lines.append(current_line)
            current_line = [(tok, atr)]
            current_len  = tok_len
        else:
            current_line.append((tok, atr))
            current_len += tok_len
    if current_line:
        lines.append(current_line)

    # Trim to however many lines actually fit
    max_lines = int((Y_START - 0.08) / LINE_H)
    truncated = len(lines) > max_lines
    lines = lines[:max_lines]

    for line_idx, line_tokens in enumerate(lines):
        y = Y_START - line_idx * LINE_H
        x = X_START

        for tok, attr in line_tokens:
            tok_w = len(tok) * CHAR_W          # width of the token text
            box_w = tok_w + CHAR_W * 0.5       # a little extra so highlights touch

            color = cmap(norm(attr))

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

            x += tok_w + CHAR_W   # token width + one space gap

    if truncated:
        trunc_y = Y_START - len(lines) * LINE_H - 0.01
        ax.text(X_START, trunc_y, "... (truncated)",
                transform=ax.transAxes, fontsize=7, color="#888888")

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.03, pad=0.01, aspect=50)
    cbar.set_label("<-- away from predicted class          towards predicted class -->",
                   fontsize=7)
    cbar.ax.tick_params(labelsize=6)


# ── Stitch all 3 panels into one PNG ─────────────────────────────────────────
def save_combined(all_model_data, sample_idx):
    """
    all_model_data: list of (tokens, attributions, version, pred_class, true_class)
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

    for v in MODEL_VERSIONS:
        load_model(v)

    found   = 0
    row_idx = 0

    while found < MAX_SAMPLES and row_idx < len(df):
        text  = df.iloc[row_idx]["paragraph"]
        label = int(df.iloc[row_idx]["label"])

        preds = {}
        for v in MODEL_VERSIONS:
            inp, _    = build_input_ref(text, v)
            preds[v], _ = predict(inp, v)

        bottleneck_ok = preds.get("bottleneckBERT") == label
        others_wrong  = all(preds[v] != label for v in MODEL_VERSIONS if v != "bottleneckBERT")

        if bottleneck_ok and others_wrong:
            found += 1
            print(f"\n-- Sample {found} (df row {row_idx})  label={label} --")

            panel_data = []
            for v in MODEL_VERSIONS:
                inp, _        = build_input_ref(text, v)
                pred_cls, _   = predict(inp, v)
                tokens, attrs = get_attributions(text, label, v)
                panel_data.append((tokens, attrs, v, pred_cls, label))
                print(f"  {v:20s}  pred={pred_cls}")

            fpath = save_combined(panel_data, found)
            print(f"  -> {fpath}")

        row_idx += 1

    print(f"\nDone. Saved {found} samples to '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()