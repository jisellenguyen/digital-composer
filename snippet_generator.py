"""
snippet_generator.py

The "tool" version of the Digital Composer. Each call to generate_snippet()
produces a fresh 10-second MIDI snippet with parameters YOU control:

    - key         which key (C, G, F#, Am, etc.)
    - mode        major or minor
    - tempo       beats per minute
    - variability how wildly the melody jumps around (0.0 = stuck, 1.0 = chaos)
    - voices      1 = monophonic, 2-4 = polyphonic harmony
    - dynamics    "soft" / "medium" / "loud" / "crescendo" / "diminuendo"
    - note_density how many notes per beat (rhythm variation, not all quarters!)

The MLP from train_and_generate.py is the "brain"; the parameters here shape
its output without retraining. This is the tool-vs-script distinction your
professor was pointing at — same model, lots of controllable behavior.

Usage:
    from snippet_generator import SnippetGenerator
    gen = SnippetGenerator(model_path="model.pkl")
    gen.generate_snippet(key="Am", tempo=140, variability=0.3, voices=2,
                         output_path="my_snippet.mid")
"""

import os
import pickle
import random
import numpy as np
import pretty_midi


# ---------------------------------------------------------------------------
# Music theory helpers
# ---------------------------------------------------------------------------

# Semitone offsets from the tonic for major/minor scales
SCALE_INTERVALS = {
    "major":     [0, 2, 4, 5, 7, 9, 11],   # whole-whole-half-whole-whole-whole-half
    "minor":     [0, 2, 3, 5, 7, 8, 10],   # natural minor
    "dorian":    [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":[0, 2, 4, 5, 7, 9, 10],
    "pentatonic":[0, 2, 4, 7, 9],          # 5-note, very forgiving
}

# MIDI pitch class for each key name (C=0, C#=1, ..., B=11)
KEY_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
    # also accept lower-case / minor-style names like "Am" → A
    "Am": 9, "Bm": 11, "Cm": 0, "Dm": 2, "Em": 4, "Fm": 5, "Gm": 7,
}

# Triad intervals for building chords from a scale degree
TRIAD_OFFSETS = [0, 4, 7]   # root, major third, fifth (we'll adjust for minor)


def build_scale(key="C", mode="major", low=48, high=84):
    """Return all MIDI pitches in the chosen key/mode within [low, high]."""
    if key not in KEY_TO_PC:
        raise ValueError(f"Unknown key '{key}'. Try C, D, Eb, F#, Am, etc.")
    if mode not in SCALE_INTERVALS:
        raise ValueError(f"Unknown mode '{mode}'. Try major, minor, dorian, "
                         f"mixolydian, pentatonic.")
    tonic_pc = KEY_TO_PC[key]
    intervals = SCALE_INTERVALS[mode]
    pitches = []
    for octave_root in range(0, 128, 12):
        for interval in intervals:
            p = octave_root + (tonic_pc + interval) % 12
            # Make sure octave is right (octave_root sets the base)
            p = octave_root + tonic_pc + interval
            if low <= p <= high:
                pitches.append(p)
    return sorted(set(pitches))


def snap_to_scale(pitch, scale_pitches):
    """Find the scale pitch closest to `pitch`. Used to force the MLP's
    raw output into the requested key."""
    return min(scale_pitches, key=lambda p: abs(p - pitch))


def velocity_curve(n_notes, dynamics="medium"):
    """Return a list of MIDI velocities (1-127) shaped by the dynamics flag."""
    if dynamics == "soft":
        return [50] * n_notes
    if dynamics == "loud":
        return [110] * n_notes
    if dynamics == "crescendo":
        return [int(40 + (90 - 40) * i / max(1, n_notes - 1))
                for i in range(n_notes)]
    if dynamics == "diminuendo":
        return [int(110 - (110 - 50) * i / max(1, n_notes - 1))
                for i in range(n_notes)]
    return [80] * n_notes  # "medium" default


def rhythm_pattern(beats, note_density=1.0, rng=None):
    """Generate note durations (in beats) summing to ~`beats`. Higher density
    means more (shorter) notes. Mixes quarters/eighths/halves so it doesn't
    sound like all quarter notes."""
    rng = rng or random.Random()
    # Available durations weighted by density
    if note_density <= 0.5:
        choices = [(2.0, 0.3), (1.0, 0.5), (0.5, 0.2)]   # mostly halves/quarters
    elif note_density <= 1.0:
        choices = [(1.0, 0.5), (0.5, 0.4), (0.25, 0.1)]  # quarters/eighths
    elif note_density <= 1.5:
        choices = [(0.5, 0.5), (0.25, 0.4), (1.0, 0.1)]  # eighths/sixteenths
    else:
        choices = [(0.25, 0.7), (0.5, 0.2), (0.125, 0.1)]  # very busy

    durations = []
    total = 0.0
    durs, weights = zip(*choices)
    while total < beats:
        d = rng.choices(durs, weights=weights)[0]
        if total + d > beats:
            d = beats - total
        durations.append(d)
        total += d
    return durations


# ---------------------------------------------------------------------------
# The main tool
# ---------------------------------------------------------------------------

class SnippetGenerator:
    """Wraps a trained MLP and exposes a controllable generation function."""

    def __init__(self, model=None, model_path=None, window_size=4):
        if model is not None:
            self.model = model
        elif model_path is not None and os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            raise ValueError("Provide either `model` or a valid `model_path`.")
        self.window_size = window_size

    def _sample_next(self, window, scale_pitches, variability, rng):
        """Predict next pitch from MLP, snap to requested scale, and apply
        variability control:
          - variability=0   → always pick argmax (most predictable)
          - variability=1   → uniform random over top candidates (chaotic)
        """
        probs = self.model.predict_proba(np.asarray(window).reshape(1, -1))[0]
        classes = self.model.classes_

        # Map variability to (top_k, temperature)
        # Low variability  → small k, low temp (sharp, predictable)
        # High variability → big k, high temp (flat, varied)
        top_k = max(2, int(2 + variability * 18))         # 2 → 20
        temperature = 0.5 + variability * 1.5             # 0.5 → 2.0

        # Apply temperature
        logp = np.log(probs + 1e-12) / temperature
        probs = np.exp(logp - logp.max())
        probs = probs / probs.sum()

        # Top-k filter
        k = min(top_k, len(probs))
        top_idx = np.argpartition(probs, -k)[-k:]
        top_probs = probs[top_idx] / probs[top_idx].sum()
        chosen = rng.choice(len(top_idx), p=top_probs)
        raw_pitch = int(classes[top_idx[chosen]])

        # Force into the requested key
        return snap_to_scale(raw_pitch, scale_pitches)

    def _generate_melody_pitches(self, n_notes, scale_pitches, variability, rng):
        """Generate `n_notes` pitches using the MLP, in the requested scale."""
        # Seed: pick a random pitch from the middle of the scale
        mid = scale_pitches[len(scale_pitches) // 2]
        seed = [mid] * self.window_size
        out = list(seed)
        for _ in range(n_notes):
            nxt = self._sample_next(out[-self.window_size:],
                                     scale_pitches, variability, rng)
            out.append(nxt)
        return out[self.window_size:]   # drop the seed

    def generate_snippet(
        self,
        duration=10.0,
        key="C",
        mode="major",
        tempo=120,
        variability=0.5,
        voices=1,
        dynamics="medium",
        note_density=1.0,
        output_path="snippet.mid",
        seed=None,
    ):
        """Generate one ~`duration`-second MIDI snippet with full parameter
        control. Returns the output path.

        Parameters
        ----------
        duration : float       seconds (target — actual duration may vary slightly)
        key      : str         "C", "F#", "Bb", "Am", etc.
        mode     : str         "major", "minor", "dorian", "mixolydian", "pentatonic"
        tempo    : int         beats per minute
        variability : float    0.0 (predictable) to 1.0 (chaotic)
        voices   : int         1 = monophonic, 2-4 = polyphonic harmony
        dynamics : str         "soft" / "medium" / "loud" / "crescendo" / "diminuendo"
        note_density : float   ~0.5 (sparse halves) to 2.0 (busy 16ths)
        """
        if seed is not None:
            rng = random.Random(seed)
            np_rng = np.random.default_rng(seed)
        else:
            rng = random.Random()
            np_rng = np.random.default_rng()

        # Convert duration to beats
        beats = duration * tempo / 60.0
        seconds_per_beat = 60.0 / tempo

        # Build the scale we'll constrain everything to
        scale_pitches = build_scale(key=key, mode=mode)

        # Generate the rhythm (list of durations in beats)
        durations_beats = rhythm_pattern(beats, note_density, rng)
        n_notes = len(durations_beats)

        # Generate the melody pitches via MLP
        # (Use np_rng-driven choices internally; rng kept for rhythm.)
        # We pass np_rng's bit_generator to a fresh wrapper for sampling
        class _NpRng:
            def __init__(self, r): self.r = r
            def choice(self, n, p): return int(self.r.choice(n, p=p))
        sampler_rng = _NpRng(np_rng)

        melody = self._generate_melody_pitches(
            n_notes, scale_pitches, variability, sampler_rng
        )

        # Build the dynamics envelope
        velocities = velocity_curve(n_notes, dynamics)

        # Build the MIDI
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        lead = pretty_midi.Instrument(program=0, name="Lead")

        # If polyphonic, also build harmony tracks
        harmony_tracks = []
        for v in range(voices - 1):
            harmony_tracks.append(
                pretty_midi.Instrument(program=0, name=f"Harmony_{v+1}")
            )

        # Place the notes
        t = 0.0
        for i, (pitch, dur_beats, vel) in enumerate(
                zip(melody, durations_beats, velocities)):
            dur_sec = dur_beats * seconds_per_beat
            lead.notes.append(pretty_midi.Note(
                velocity=vel, pitch=int(pitch),
                start=t, end=t + dur_sec * 0.95   # slight gap between notes
            ))

            # Harmony voices: stack thirds and fifths from the scale
            for v_idx, h_track in enumerate(harmony_tracks):
                # Find pitch's position in the scale, then offset by 2 or 4
                # scale steps to get a third or fifth
                if pitch in scale_pitches:
                    sp_idx = scale_pitches.index(pitch)
                else:
                    sp_idx = min(range(len(scale_pitches)),
                                 key=lambda j: abs(scale_pitches[j] - pitch))
                offset = (v_idx + 1) * 2   # 2 = third, 4 = fifth
                h_idx = max(0, min(len(scale_pitches) - 1, sp_idx - offset))
                h_pitch = scale_pitches[h_idx]
                h_track.notes.append(pretty_midi.Note(
                    velocity=max(30, vel - 20),   # quieter than lead
                    pitch=int(h_pitch),
                    start=t, end=t + dur_sec * 0.95
                ))

            t += dur_sec

        pm.instruments.append(lead)
        for h in harmony_tracks:
            pm.instruments.append(h)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pm.write(output_path)
        return output_path


# ---------------------------------------------------------------------------
# Demo: show off the parameter control
# ---------------------------------------------------------------------------

def demo(model_path="model.pkl", out_dir="output/snippets"):
    """Generate a 'mix and match' demo of 5 different parameter combinations
    so you can show off the control your tool gives you."""
    gen = SnippetGenerator(model_path=model_path)
    os.makedirs(out_dir, exist_ok=True)

    settings = [
        dict(name="01_calm_major",
             key="C", mode="major", tempo=80, variability=0.2,
             voices=1, dynamics="soft", note_density=0.7),
        dict(name="02_chaotic_minor",
             key="A", mode="minor", tempo=140, variability=0.9,
             voices=1, dynamics="loud", note_density=1.5),
        dict(name="03_polyphonic_pentatonic",
             key="G", mode="pentatonic", tempo=110, variability=0.4,
             voices=3, dynamics="medium", note_density=1.0),
        dict(name="04_crescendo_dorian",
             key="D", mode="dorian", tempo=100, variability=0.5,
             voices=2, dynamics="crescendo", note_density=1.2),
        dict(name="05_dramatic_minor_dense",
             key="E", mode="minor", tempo=160, variability=0.7,
             voices=2, dynamics="loud", note_density=1.8),
    ]

    print(f"Generating {len(settings)} snippets in {out_dir}/")
    for s in settings:
        name = s.pop("name")
        path = gen.generate_snippet(
            duration=10.0, output_path=os.path.join(out_dir, f"{name}.mid"),
            seed=42, **s
        )
        print(f"  {path}")
    print("\nDone. Compare them — same model, completely different output.")


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model.pkl"
    demo(model_path=model_path)
