"""
CLIP Delta-Z Decomposition  (Google 10k word list)
---------------------------------------------------
1. Fetch the Google 10,000-word list from GitHub.
2. Encode two images with CLIP → raw embedding vectors.
3. Compute ΔZ = embed(img2) - embed(img1).
4. Build a word dictionary: word → CLIP text unit vector  (batched).
5. Express ΔZ as a linear combination of those unit vectors
   (coefficients = dot products of ΔZ with each unit vector).
"""

import urllib.request
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ── Google 10k word list ───────────────────────────────────────────────────────

GOOGLE_10K_URL = (
    "https://raw.githubusercontent.com/first20hours/google-10000-english/"
    "master/google-10000-english-no-swears.txt"
)

def fetch_google_10k(url: str = GOOGLE_10K_URL) -> list[str]:
    """
    Download the Google 10k word list, strip proper nouns (names, places, brands),
    and return only common English words that appear in WordNet.
    """
    import nltk
    nltk.download("wordnet", quiet=True)
    from nltk.corpus import wordnet as wn

    # Build a set of all lowercase lemma names that appear in WordNet
    wordnet_words = {
        lemma.name().lower().replace("_", " ")
        for synset in wn.all_synsets()
        for lemma in synset.lemmas()
    }

    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8")

    raw = [
        line.strip().lower()
        for line in text.splitlines()
        if line.strip() and " " not in line.strip()
    ]

    # Keep only words that WordNet knows — this drops proper nouns cleanly
    filtered = [w for w in raw if w in wordnet_words]

    print(f"  Fetched {len(raw)} words, kept {len(filtered)} after removing proper nouns.")
    return filtered


# ── Model loading ──────────────────────────────────────────────────────────────

def load_clip(model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor, device


# ── Image encoding ─────────────────────────────────────────────────────────────

def encode_image(image_path: str, model, processor, device) -> np.ndarray:
    """Return the raw (un-normalised) CLIP image embedding as a 1-D numpy array."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)   # (1, D)
    return features.squeeze(0).cpu().numpy()            # (D,)


# ── Text encoding (batched) ────────────────────────────────────────────────────

def build_word_dict(
    words: list[str],
    model,
    processor,
    device,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """
    Encode all words in batches and return {word: unit_vector}.
    Batching is ~100× faster than encoding one word at a time.
    """
    all_vecs = []
    for i in range(0, len(words), batch_size):
        batch = words[i : i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)           # (B, D)
        # Normalise each vector to unit length
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_vecs.append(feats.cpu().numpy())
        if (i // batch_size) % 5 == 0:
            print(f"  … encoded {min(i + batch_size, len(words))}/{len(words)} words", end="\r")

    print()  # newline after progress
    unit_vecs = np.concatenate(all_vecs, axis=0)    # (N, D)
    return dict(zip(words, unit_vecs))


# ── Greedy matching pursuit decomposition ─────────────────────────────────────

def decompose_pursuit(
    delta_z: np.ndarray,
    word_dict: dict[str, np.ndarray],
    n_steps: int = 10,
) -> list[dict]:
    """
    Greedy matching pursuit: iteratively explain the residual.

    Each step:
      1. Find the word whose unit vector has the largest |dot product| with
         the current residual.
      2. Record the coefficient (signed dot product).
      3. Subtract that word's contribution from the residual: r <- r - coeff*u

    Returns a list of step dicts, one per selected word:
      {
        "step":          int,
        "word":          str,
        "coeff":         float,   # contribution removed at this step
        "residual_norm": float,   # norm of residual after subtraction
        "explained":     float,   # fraction of original norm explained so far
      }
    """
    words  = list(word_dict.keys())
    matrix = np.stack(list(word_dict.values()))   # (N, D)
    used   = set()

    residual      = delta_z.copy().astype(np.float64)
    original_norm = np.linalg.norm(residual)
    steps         = []

    for step in range(1, n_steps + 1):
        dots = matrix @ residual          # (N,) — project residual onto all words
        for idx in used:
            dots[idx] = 0.0               # mask already-used words

        best_idx  = int(np.argmax(np.abs(dots)))
        best_word = words[best_idx]
        coeff     = float(dots[best_idx])

        residual -= coeff * matrix[best_idx]   # subtract this component

        residual_norm = float(np.linalg.norm(residual))
        explained     = 1.0 - residual_norm / original_norm

        steps.append({
            "step":          step,
            "word":          best_word,
            "coeff":         coeff,
            "residual_norm": residual_norm,
            "explained":     explained,
        })
        used.add(best_idx)

    return steps


# ── (kept for reference) flat dot-product decompose ───────────────────────────

def decompose(delta_z: np.ndarray, word_dict: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Express delta_z as a linear combination of the word unit vectors.

    coefficient_i  =  ΔZ · û_i

    Positive coefficient  → image2 is "more" of that concept than image1.
    Negative coefficient  → image2 is "less" of that concept than image1.

    Uses a vectorised matrix multiply for speed over the full 10k vocabulary.
    """
    words = list(word_dict.keys())
    matrix = np.stack(list(word_dict.values()))     # (N, D)
    coeffs = matrix @ delta_z                       # (N,)  — all dot products at once
    return dict(zip(words, coeffs.tolist()))


# ── Main ───────────────────────────────────────────────────────────────────────

def main(
    image_path_1: str,
    image_path_2: str,
    n_steps: int = 10,
):
    print("Fetching Google 10k word list …")
    words = fetch_google_10k()

    print("\nLoading CLIP …")
    model, processor, device = load_clip()
    print(f"  device: {device}")

    # Raw image embeddings (not normalised)
    print("\nEncoding images …")
    z1 = encode_image(image_path_1, model, processor, device)
    z2 = encode_image(image_path_2, model, processor, device)

    # ΔZ in raw embedding space
    delta_z = z2 - z1
    print(f"  ΔZ  shape : {delta_z.shape}")
    print(f"  ΔZ  norm  : {np.linalg.norm(delta_z):.4f}")

    # Word dictionary (batched)
    print(f"\nBuilding word dictionary for {len(words)} words …")
    word_dict = build_word_dict(words, model, processor, device)

    # Greedy pursuit decomposition
    print(f"\nRunning matching pursuit ({n_steps} steps) …\n")
    steps = decompose_pursuit(delta_z, word_dict, n_steps=n_steps)

    # Human-readable equation:  ΔZ ≈ +3.21·"sunny" - 1.84·"dark" + …
    terms = []
    for s in steps:
        sign = "+" if s["coeff"] >= 0 else "-"
        terms.append(f'{sign} {abs(s["coeff"]):.3f}·"{s["word"]}"')
    print("── Decomposition ──")
    print("  ΔZ ≈ " + " ".join(terms))
    print()

    # Step-by-step table
    print(f"  {'step':<6} {'word':<20} {'coeff':>10}  {'‖residual‖':>12}  {'explained':>10}")
    print("  " + "─" * 64)
    for s in steps:
        bar = "▲" if s["coeff"] > 0 else "▼"
        print(
            f"  {s['step']:<6} {s['word']:<20} {s['coeff']:>+10.4f}  "
            f"{s['residual_norm']:>12.4f}  {s['explained']:>9.1%}  {bar}"
        )

    return delta_z, word_dict, steps


# ── Example usage ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    delta_z, word_dict, coefficients = main(
        image_path_1="samples/test_lion_1.png",   # ← replace with your paths
        image_path_2="samples/test_lion_2.jpg",
        n_steps=10,
    )