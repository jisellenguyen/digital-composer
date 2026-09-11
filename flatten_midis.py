"""
flatten_midis.py

Walks a folder of MIDI files and produces a single CSV of rolling windows
suitable for scikit-learn. Each row is:

    [pitch_{t-W+1}, ..., pitch_{t-1}, pitch_t,  target_pitch_{t+1}]

Design choices (these are the kinds of things worth mentioning in your
presentation — they're small but they matter):

  1) We only look at pitch, not rhythm. Extending to duration is an obvious
     next step; keeping it pitch-only first isolates one variable.

  2) We use the monophonic melody line extracted from whichever instrument
     has the most notes. This is a rough heuristic but works well on NES /
     classical single-lead data.

  3) We optionally transpose every file to C (using pretty_midi's key
     estimation). This is a huge simplification: the model learns ONE key
     instead of 12, which means much less data is needed.

  4) Windows never cross file boundaries (otherwise we'd have "notes" that
     don't actually follow each other).
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import pretty_midi


def extract_melody(pm):
    """Return a list of MIDI pitches from the instrument with the most notes.
    We sort its notes by start time and drop any simultaneous duplicates."""
    if not pm.instruments:
        return []
    # Skip drum tracks — their "pitches" aren't pitches, they're drum IDs
    non_drum = [inst for inst in pm.instruments if not inst.is_drum]
    if not non_drum:
        return []
    melody_inst = max(non_drum, key=lambda inst: len(inst.notes))
    notes = sorted(melody_inst.notes, key=lambda n: n.start)

    pitches = []
    last_start = -1.0
    for n in notes:
        # Skip chord stacks — keep only the first note at each onset
        if n.start > last_start + 1e-4:
            pitches.append(n.pitch)
            last_start = n.start
    return pitches


def transpose_to_c(pitches, pm):
    """Estimate the key, then shift everything so tonic = C (MIDI 60 mod 12).
    If we can't estimate, leave it alone."""
    if not pitches:
        return pitches
    try:
        # pretty_midi doesn't ship key estimation; use a simple pitch-class
        # histogram heuristic instead. Most common pitch class ≈ tonic.
        pc_counts = np.bincount([p % 12 for p in pitches], minlength=12)
        tonic_pc = int(np.argmax(pc_counts))
        shift = (0 - tonic_pc) % 12
        # Keep shift in [-6, +6] so we don't move by a whole octave
        if shift > 6:
            shift -= 12
        return [p + shift for p in pitches]
    except Exception:
        return pitches


def make_windows(pitches, window_size):
    """Rolling windows of `window_size` features + 1 target."""
    rows = []
    for i in range(len(pitches) - window_size):
        rows.append(pitches[i : i + window_size] + [pitches[i + window_size]])
    return rows


def flatten(midi_dir, out_csv, window_size=4, transpose=True, verbose=True):
    paths = sorted(glob.glob(os.path.join(midi_dir, "*.mid")) +
                   glob.glob(os.path.join(midi_dir, "*.midi")))
    if not paths:
        raise FileNotFoundError(f"No .mid/.midi files found in {midi_dir}")

    all_rows = []
    n_ok, n_skip = 0, 0
    for path in paths:
        try:
            pm = pretty_midi.PrettyMIDI(path)
        except Exception as e:
            if verbose:
                print(f"  skip (parse error): {os.path.basename(path)} -- {e}")
            n_skip += 1
            continue

        pitches = extract_melody(pm)
        if len(pitches) < window_size + 1:
            n_skip += 1
            continue
        if transpose:
            pitches = transpose_to_c(pitches, pm)

        rows = make_windows(pitches, window_size)
        all_rows.extend(rows)
        n_ok += 1

    cols = [f"n_{i}" for i in range(window_size)] + ["target"]
    df = pd.DataFrame(all_rows, columns=cols)
    df.to_csv(out_csv, index=False)

    if verbose:
        print(f"\nProcessed {n_ok} files ({n_skip} skipped)")
        print(f"Wrote {len(df):,} training windows to {out_csv}")
        print(f"Unique target pitches: {df['target'].nunique()}")
        print(f"Pitch range: {df['target'].min()}–{df['target'].max()}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_dir", default="data/sample_midis")
    ap.add_argument("--out_csv", default="data/windows.csv")
    ap.add_argument("--window_size", type=int, default=4)
    ap.add_argument("--no_transpose", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    flatten(args.midi_dir, args.out_csv,
            window_size=args.window_size,
            transpose=not args.no_transpose)
