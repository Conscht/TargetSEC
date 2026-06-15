"""
Style embedding space — before/after fine-tuning, coloured by arousal class (1–7).

3-panel figure:
  Left   : Old encoder (LibriTTS pretrained) — coloured by arousal class
  Centre : Fine-tuned encoder (MSP-Podcast)  — coloured by arousal class
  Right  : Both in one space with grey arrows showing per-utterance shift,
           points coloured by arousal class

Arousal class is derived by passing pre-computed 1024-dim emotion hidden
states (emotion_embeddings.pth) through the regression head → continuous
arousal in [0,1] → discretised to class 1–7.

Generates both UMAP and PCA versions. Embeddings are cached to
style_shift_cache.npz to avoid recomputing on re-runs.

Output: style_shift_plasma_umap.png  and  style_shift_plasma_pca.png
"""
import ast
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.cm as cm

from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from torch import nn

from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config
from src.dataset import test_create_data_loader

# ── Paths ──────────────────────────────────────────────────────────────────
PRETRAINED_PATH = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/"
    "Audio/MSP-Podcast-1.10/pre-trained_models/pre-trained_style"
)
FINETUNE_CHECKPOINT = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/"
    "checkpoints_synthesizer_finetune/"
    "synthesizer_finetune-06-13_18-22-18-epoch=48-val_loss=16.24.ckpt"
)
EMO_EMBEDDINGS_PATH = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/"
    "Audio/MSP-Podcast-1.10/emotion_embeddings.pth"
)
TEST1_MANIFEST = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/"
    "Audio/MSP-Podcast-1.10/hubert-km100/parsed_with_spkrEmbeds/test1.txt"
)
EMO_MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
CACHE_PATH     = "style_shift_cache.npz"

N_SAMPLES  = 16903
SEG_FRAMES = 156
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Discrete 7-class colormap: blue (calm) → red (activated)
_CMAP7   = matplotlib.colormaps["plasma"].resampled(7)
_BOUNDS7 = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
_NORM7   = mcolors.BoundaryNorm(_BOUNDS7, _CMAP7.N)


# ── Emotion regression head ───────────────────────────────────────────────
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense    = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout  = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2   = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0].mean(dim=1)
        return hidden, self.classifier(hidden)


# ── Helpers ───────────────────────────────────────────────────────────────
def load_finetuned_encoder(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    keys = {k[len("style_encoder."):]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("style_encoder.")}
    enc = MelStyleEncoder(style_config)
    enc.load_state_dict(keys, strict=True)
    return enc


def center_window(mel, seg=SEG_FRAMES):
    T = mel.shape[1]
    c = T // 2
    s = max(0, c - seg // 2)
    return mel[:, s:s + seg, :]


def get_arousal_classes(basenames, emo_embs_dict, regression_head):
    """
    Pass pre-computed 1024-dim hidden states through regression head.
    Maps continuous arousal [0, 1] → discrete class 1–7.
    """
    raw = []
    regression_head.eval().to(DEVICE)
    with torch.no_grad():
        for bn in basenames:
            if bn and bn in emo_embs_dict:
                h = emo_embs_dict[bn].float().to(DEVICE)
                if h.dim() == 3:
                    h = h.squeeze(1)
                logits = regression_head(h)
                raw.append(logits[0, 0].item())
            else:
                raw.append(float("nan"))
    arousal_cont = np.array(raw)
    # Map [0,1] → [1,7]: round(a*6+1), clip to [1,7]
    classes = np.clip(np.round(arousal_cont * 6 + 1).astype(float), 1, 7)
    classes[np.isnan(arousal_cont)] = np.nan
    return arousal_cont, classes


def reduce_2d_umap(combined):
    import umap as umap_lib
    print("  Fitting UMAP (n_neighbors=30, min_dist=0.1)...")
    r = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    return r.fit_transform(combined)


def reduce_2d_pca(combined):
    from sklearn.decomposition import PCA
    print("  Fitting PCA...")
    r = PCA(n_components=2, random_state=42)
    return r.fit_transform(combined)


def make_plot(old_2d, new_2d, classes, method, out_path):
    scatter_kw = dict(s=8, alpha=0.7, linewidths=0, cmap=_CMAP7, norm=_NORM7)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"Style Embedding Space — Before vs. After MSP-Podcast Fine-Tuning ({method})",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── Left: old encoder ─────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(old_2d[:, 0], old_2d[:, 1], c=classes, **scatter_kw)
    ax.set_title("Original encoder\n(LibriTTS pretrained)", fontsize=11)
    ax.set_xlabel(f"{method} 1"); ax.set_ylabel(f"{method} 2")
    ax.set_aspect("equal", adjustable="datalim")

    # ── Centre: fine-tuned encoder ────────────────────────────────────────
    ax = axes[1]
    ax.scatter(new_2d[:, 0], new_2d[:, 1], c=classes, **scatter_kw)
    ax.set_title("Fine-tuned encoder\n(MSP-Podcast)", fontsize=11)
    ax.set_xlabel(f"{method} 1")
    ax.set_aspect("equal", adjustable="datalim")

    # ── Right: shift arrows ───────────────────────────────────────────────
    ax = axes[2]
    # Subsample arrows for visibility — 2000 random pairs with clear arrowheads
    rng = np.random.default_rng(42)
    idx = rng.choice(len(old_2d), size=min(2000, len(old_2d)), replace=False)
    for i in idx:
        ax.annotate(
            "", xy=new_2d[i], xytext=old_2d[i],
            arrowprops=dict(arrowstyle="-|>", color="grey", lw=0.9, alpha=0.35,
                            mutation_scale=6),
        )
    ax.scatter(old_2d[:, 0], old_2d[:, 1], c=classes,
               marker="o", s=8, alpha=0.6, linewidths=0,
               cmap=_CMAP7, norm=_NORM7, zorder=3)
    ax.scatter(new_2d[:, 0], new_2d[:, 1], c=classes,
               marker="^", s=8, alpha=0.6, linewidths=0,
               cmap=_CMAP7, norm=_NORM7, zorder=4)
    ax.set_title("Shift per utterance\n(○ original → △ fine-tuned)", fontsize=11)
    ax.set_xlabel(f"{method} 1")
    ax.set_aspect("equal", adjustable="datalim")
    shape_handles = [
        plt.Line2D([0], [0], marker="o", color="grey", linestyle="none",
                   markersize=6, label="Original"),
        plt.Line2D([0], [0], marker="^", color="grey", linestyle="none",
                   markersize=6, label="Fine-tuned"),
    ]
    ax.legend(handles=shape_handles, fontsize=9, loc="upper right")

    # ── Single large colorbar to the right of all panels ──────────────────
    fig.subplots_adjust(right=0.87)
    cax = fig.add_axes([0.89, 0.12, 0.018, 0.76])
    cb = fig.colorbar(cm.ScalarMappable(norm=_NORM7, cmap=_CMAP7),
                      cax=cax, ticks=range(1, 8))
    cb.set_label("Arousal class", fontsize=11, labelpad=10)
    cb.ax.set_yticklabels([str(c) for c in range(1, 8)], fontsize=10)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    # ── Load or compute embeddings ─────────────────────────────────────────
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached embeddings from {CACHE_PATH} ...")
        cache = np.load(CACHE_PATH)
        old_embs = cache["old_embs"]
        new_embs = cache["new_embs"]
        basenames_arr = cache["basenames"]
        basenames = list(basenames_arr)
    else:
        print("Loading style encoders...")
        old_enc = MelStyleEncoder(style_config)
        old_enc.load_state_dict(torch.load(PRETRAINED_PATH, map_location="cpu"))
        new_enc = load_finetuned_encoder(FINETUNE_CHECKPOINT)

        print("Reading manifest basenames...")
        basenames = []
        with open(TEST1_MANIFEST) as f:
            for line in f:
                data = ast.literal_eval(line.strip())
                basenames.append(os.path.basename(data["audio"]))
        basenames = basenames[:N_SAMPLES]

        loader = test_create_data_loader(batch_size=1)
        old_embs_list, new_embs_list = [], []
        old_enc.eval().to(DEVICE)
        new_enc.eval().to(DEVICE)

        print(f"Collecting style embeddings from {N_SAMPLES} utterances...")
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= N_SAMPLES:
                    break
                mel = batch["mel_spectrogram"].to(DEVICE)
                win = center_window(mel)
                old_embs_list.append(old_enc(win).squeeze(0).cpu())
                new_embs_list.append(new_enc(win).squeeze(0).cpu())
                if (i + 1) % 1000 == 0:
                    print(f"  {i+1}/{N_SAMPLES}")

        old_embs = torch.stack(old_embs_list).numpy()
        new_embs = torch.stack(new_embs_list).numpy()
        np.savez(CACHE_PATH, old_embs=old_embs, new_embs=new_embs,
                 basenames=np.array(basenames))
        print(f"Embeddings cached → {CACHE_PATH}")

    # ── Arousal classes ────────────────────────────────────────────────────
    print("Loading emotion regression head for arousal labels...")
    emo_model = EmotionModel.from_pretrained(EMO_MODEL_NAME)
    regression_head = emo_model.classifier
    del emo_model

    print("Loading pre-computed emotion embeddings...")
    emo_embs = torch.load(EMO_EMBEDDINGS_PATH, map_location="cpu")

    print("Computing arousal classes...")
    arousal_cont, classes = get_arousal_classes(basenames, emo_embs, regression_head)
    valid = ~np.isnan(classes)
    print(f"  Classes found for {valid.sum()}/{N_SAMPLES} utterances")
    for c in range(1, 8):
        n = (classes == c).sum()
        print(f"    Class {c}: {n} utterances ({100*n/N_SAMPLES:.1f}%)")

    # ── 2-D projections ────────────────────────────────────────────────────
    combined = np.vstack([old_embs, new_embs])
    n = N_SAMPLES

    # UMAP
    try:
        umap_proj = reduce_2d_umap(combined)
        make_plot(umap_proj[:n], umap_proj[n:], classes, "UMAP", "style_shift_plasma_umap.png")
        shift_u = np.linalg.norm(umap_proj[n:] - umap_proj[:n], axis=1)
        print(f"UMAP shift: mean={shift_u.mean():.3f}  median={np.median(shift_u):.3f}"
              f"  std={shift_u.std():.3f}  max={shift_u.max():.3f}")
    except ImportError:
        print("umap-learn not installed — skipping UMAP.")

    # PCA
    pca_proj = reduce_2d_pca(combined)
    make_plot(pca_proj[:n], pca_proj[n:], classes, "PCA", "style_shift_plasma_pca.png")
    shift_p = np.linalg.norm(pca_proj[n:] - pca_proj[:n], axis=1)
    print(f"PCA  shift: mean={shift_p.mean():.3f}  median={np.median(shift_p):.3f}"
          f"  std={shift_p.std():.3f}  max={shift_p.max():.3f}")


if __name__ == "__main__":
    main()
