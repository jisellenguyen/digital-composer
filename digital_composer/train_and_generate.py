"""
train_and_generate.py

End-to-end pipeline:
  1) Load the flattened CSV
  2) Train/test split
  3) Train the MLPComposer
  4) Measure next-note accuracy on held-out data for each composer
  5) Generate a MIDI file from each composer so you can LISTEN to the results
  6) Save the trained model to disk so snippet_generator.py can use it
     without retraining

The accuracy numbers are great for the presentation — but the real test is
whether the MIDI files sound different from each other. They will.
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from composers import (
    RandomComposer, ScaleRandomComposer, PiComposer, MLPComposer,
    pitches_to_midi,
)


def evaluate_next_note_accuracy(composer, X_test, y_test, is_mlp=False):
    """How often does the composer's 'guess' for the next note match the
    true next note in the test set? For non-MLP composers this is really a
    sanity check — they don't actually look at the context, so random/scale
    will score near chance, and that's exactly the point of the baseline."""
    if is_mlp:
        y_pred = composer.model.predict(X_test)
        return float(np.mean(y_pred == y_test))

    # For rule-based composers: generate a single note without context and
    # ask how often it matches. (This intentionally under-measures them —
    # they're blind baselines.)
    correct = 0
    for true in y_test:
        guess = composer.generate(n_notes=1)[0]
        if guess == true:
            correct += 1
    return correct / len(y_test)


def top_k_accuracy(model, X_test, y_test, k=3):
    """Top-k accuracy: does the true note appear in the model's top-k
    predictions? This is a more forgiving (and more musical) metric —
    several notes are often 'reasonable' continuations."""
    probs = model.predict_proba(X_test)
    classes = model.classes_
    top_k_idx = np.argsort(probs, axis=1)[:, -k:]
    top_k_classes = classes[top_k_idx]
    hits = [y in row for y, row in zip(y_test, top_k_classes)]
    return float(np.mean(hits))


def main(csv_path, out_dir, window_size, n_gen_notes):
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1) Load data ------------------------------------------------------
    df = pd.read_csv(csv_path)
    assert df.shape[1] == window_size + 1, (
        f"CSV has {df.shape[1]} columns but window_size={window_size} "
        f"implies {window_size + 1}. Re-run flatten_midis.py with matching "
        f"--window_size."
    )
    X = df.iloc[:, :window_size].values
    y = df.iloc[:, window_size].values
    print(f"Loaded {len(df):,} windows of size {window_size}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # ---- 2) Instantiate all four composers ---------------------------------
    random_c = RandomComposer()
    scale_c = ScaleRandomComposer()
    pi_c = PiComposer()
    mlp_c = MLPComposer(window_size=window_size).fit(X_tr, y_tr)

    # ---- 3) Evaluate accuracy ---------------------------------------------
    print("\n=== Next-note accuracy on held-out test set ===")
    # Baselines: shrink y_te so they don't take forever
    y_te_small = y_te[:500]
    print(f"{'Composer':<18}{'Top-1':>10}{'Top-3':>10}")
    for c in (random_c, scale_c, pi_c):
        acc = evaluate_next_note_accuracy(c, None, y_te_small)
        print(f"{c.name:<18}{acc*100:>9.2f}%{'  --':>10}")

    mlp_top1 = evaluate_next_note_accuracy(mlp_c, X_te, y_te, is_mlp=True)
    mlp_top3 = top_k_accuracy(mlp_c.model, X_te, y_te, k=3)
    print(f"{'mlp':<18}{mlp_top1*100:>9.2f}%{mlp_top3*100:>9.2f}%")

    # ---- 4) Generate one MIDI file per composer ---------------------------
    print("\n=== Generating MIDI samples ===")
    seed = list(X_tr[0])
    for c in (random_c, scale_c, pi_c, mlp_c):
        notes = c.generate(n_notes=n_gen_notes, seed_window=seed)
        path = os.path.join(out_dir, f"generated_{c.name}.mid")
        pitches_to_midi(notes, path)
        print(f"  wrote {path}  (first 12 notes: {notes[:12]})")

    print("\nDone. Open the .mid files in GarageBand / MuseScore / "
          "any DAW to listen — the difference between tiers is audible.")

    # ---- 5) Save the trained model so the snippet tool can use it --------
    model_path = os.path.join(out_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(mlp_c.model, f)
    print(f"\nSaved trained model to {model_path}")
    print("Now you can run: python3 snippet_generator.py "
          f"{model_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/windows.csv")
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--window_size", type=int, default=4)
    ap.add_argument("--n_gen_notes", type=int, default=64)
    args = ap.parse_args()
    main(args.csv, args.out_dir, args.window_size, args.n_gen_notes)
