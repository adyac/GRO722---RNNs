"""
eval_samples.py  —  Load model.pt and plot attention-weighted predictions
in the same style as the project report figures.

Usage examples
--------------
# 5 random samples, displayed on screen
python eval_samples.py

# 10 random samples saved as PNG files
python eval_samples.py --n 10 --save figures/

# specific sample indices
python eval_samples.py --idx 0 7 42

# different model / data paths
python eval_samples.py --model my_model.pt --data data_test.p
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# ── allow running from any working directory ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import HandwrittenWords
from models import trajectory2seq


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_path, dataset, device):
    model = trajectory2seq(
        hidden_dim=18,
        n_layers=1,
        symb2int=dataset.symb2int,
        int2symb=dataset.int2symb,
        dict_size=dataset.dict_size,
        device=device,
        maxlen=dataset.max_len,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# ── inference ─────────────────────────────────────────────────────────────────

def predict(model, x_tensor, dataset, device):
    """
    Run one sample through the model.
    Returns pred_str, attn_map (T × max_word_len), coords (2 × T).
    """
    x_input = x_tensor.unsqueeze(0).to(device).float()
    with torch.no_grad():
        output, _, attn = model(x_input)

    pred_indices = torch.argmax(output, dim=-1).cpu().squeeze().tolist()
    if isinstance(pred_indices, int):
        pred_indices = [pred_indices]

    # trim at <eos> (index 1)
    try:
        pred_indices = pred_indices[:pred_indices.index(1)]
    except ValueError:
        pass

    pred_str = [dataset.int2symb['word'][i] for i in pred_indices]
    attn_map = attn.squeeze(0).cpu().numpy()   # (T, max_len_word)
    coords   = x_tensor.numpy().T              # (2, T)
    return pred_str, attn_map, coords


# ── visualisation ─────────────────────────────────────────────────────────────

def plot_sample(coords, attn_map, pred_str, true_str, save_path=None):
    """
    One subplot per predicted letter.
    Gray scatter: dark dots = high attention, light dots = low attention.
    Matches the style of the islet/lynn/lela figures in the report.

    coords   : (2, T)  x/y coordinate array
    attn_map : (T, max_len_word) raw attention weights
    pred_str : list of predicted letter strings
    true_str : list of ground-truth letter strings
    save_path: if given, save PNG there instead of calling plt.show()
    """
    n_letters = max(len(pred_str), 1)
    fig, axes = plt.subplots(n_letters, 1, figsize=(6, 2 * n_letters))
    if n_letters == 1:
        axes = [axes]

    for j, letter in enumerate(pred_str):
        ax = axes[j]

        # extract and normalise attention column for letter j
        attn_j = attn_map[:, j]
        attn_j = (attn_j - attn_j.min()) / (attn_j.max() - attn_j.min() + 1e-8)
        attn_j = 1 - attn_j          # invert: high attention → dark colour

        ax.plot(coords[0], coords[1], c='lightgray', linewidth=0.8, zorder=1)
        ax.scatter(coords[0], coords[1], c=attn_j, cmap='gray',
                   vmin=0, vmax=1, s=12, zorder=2)
        ax.set_ylabel(letter, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    target_word     = ' '.join(true_str)
    prediction_word = ' '.join(pred_str) if pred_str else '(empty)'
    plt.suptitle(f"Target: {target_word}\nPrediction: {prediction_word}", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved  →  {save_path}")
    else:
        plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Evaluate a saved model.pt and plot attention predictions.")
    parser.add_argument("--model", default=os.path.join(here, "model.pt"),
                        help="Path to saved state dict  (default: model.pt)")
    parser.add_argument("--data",  default=os.path.join(here, "data_trainval.p"),
                        help="Path to dataset pickle    (default: data_trainval.p)")
    parser.add_argument("--n",     type=int, default=5,
                        help="Number of random samples  (default: 5)")
    parser.add_argument("--idx",   type=int, nargs="+",
                        help="Specific sample indices (overrides --n)")
    parser.add_argument("--save",  default=None,
                        help="Directory to save PNGs instead of displaying")
    parser.add_argument("--seed",  type=int, default=42,
                        help="Random seed for sample selection (default: 42)")
    args = parser.parse_args()

    # fall back to data_test.p if the default file is missing
    if not os.path.exists(args.data):
        alt = os.path.join(here, "data_test.p")
        if os.path.exists(alt):
            print(f"data_trainval.p not found, using {alt}")
            args.data = alt
        else:
            sys.exit(f"Error: dataset not found at {args.data}")

    if not os.path.exists(args.model):
        sys.exit(f"Error: model not found at {args.model}")

    device = torch.device("cpu")

    print(f"Dataset : {args.data}")
    dataset = HandwrittenWords(args.data)

    print(f"Model   : {args.model}")
    model = load_model(args.model, dataset, device)
    # keep maxlen consistent with the loaded dataset
    model.maxlen['handwritten'] = dataset.max_len['handwritten']

    # pick indices
    if args.idx:
        indices = args.idx
    else:
        rng     = np.random.default_rng(args.seed)
        indices = rng.choice(len(dataset), size=min(args.n, len(dataset)),
                             replace=False).tolist()

    if args.save:
        os.makedirs(args.save, exist_ok=True)

    print(f"\n{'idx':>6}  {'target':<12}  {'prediction'}")
    print("-" * 36)

    for i, idx in enumerate(indices):
        x, y = dataset[idx]

        # ground-truth word (trim padding and <eos>)
        true_indices = y.tolist()
        try:
            true_indices = true_indices[:true_indices.index(1)]
        except ValueError:
            pass
        true_str = [dataset.int2symb['word'][k] for k in true_indices]

        pred_str, attn_map, coords = predict(model, x, dataset, device)

        target_word    = ''.join(true_str)
        predicted_word = ''.join(pred_str) if pred_str else '(empty)'
        match = '✓' if target_word == predicted_word else '✗'
        print(f"{idx:>6}  {target_word:<12}  {predicted_word}  {match}")

        save_path = None
        if args.save:
            fname     = f"sample_{i:03d}_{target_word}.png"
            save_path = os.path.join(args.save, fname)

        plot_sample(coords, attn_map, pred_str, true_str, save_path=save_path)


if __name__ == "__main__":
    main()
