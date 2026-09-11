"""
composers.py

Four "Digital Composers," from dumbest to smartest, so we can directly compare
them. Each one implements the same interface:

    composer.generate(n_notes=64, seed_window=None) -> List[int]   # MIDI pitches

So swapping them in/out for evaluation is trivial.

Tiers:
    1) RandomComposer         — uniform random over the full MIDI pitch range
    2) ScaleRandomComposer    — uniform random restricted to C major
    3) PiComposer             — digits of pi mapped to scale degrees
    4) MLPComposer            — MLPClassifier trained on the flattened windows
"""

import math
import random
import numpy as np
from sklearn.neural_network import MLPClassifier


C_MAJOR_SCALE = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84]


# ------------------------------------------------------------------ Tier 1 --
class RandomComposer:
    """Baseline: uniform random pitch in a reasonable piano range.
    We expect this to sound like noise. That is the point."""
    name = "random"

    def __init__(self, low=48, high=84, seed=0):
        self.low, self.high = low, high
        self.rng = random.Random(seed)

    def generate(self, n_notes=64, seed_window=None):
        return [self.rng.randint(self.low, self.high) for _ in range(n_notes)]


# ------------------------------------------------------------------ Tier 2 --
class ScaleRandomComposer:
    """Still random, but constrained to a scale. This one change alone
    dramatically improves perceived musicality — a nice teaching moment
    about how much structure matters vs. 'intelligence'."""
    name = "scale_random"

    def __init__(self, scale=C_MAJOR_SCALE, seed=0):
        self.scale = scale
        self.rng = random.Random(seed)

    def generate(self, n_notes=64, seed_window=None):
        return [self.rng.choice(self.scale) for _ in range(n_notes)]


# ------------------------------------------------------------------ Tier 3 --
class PiComposer:
    """Map digits of pi (0-9) to scale degrees. Deterministic, structured,
    but non-learned. Often sounds surprisingly listenable because consecutive
    pi digits have no long-range pattern but feel 'wandering'."""
    name = "pi"

    # ~200 digits of pi — plenty for any reasonable n_notes
    PI_DIGITS = (
        "141592653589793238462643383279502884197169399375105820974944"
        "592307816406286208998628034825342117067982148086513282306647"
        "093844609550582231725359408128481117450284102701938521105559"
        "644622948954930381964428810975665933446128475648233786783165"
    )

    def __init__(self, scale=C_MAJOR_SCALE):
        # Map digit 0–9 → a scale pitch. We pick 10 notes from the scale.
        self.digit_to_pitch = [scale[i % len(scale)] for i in range(10)]

    def generate(self, n_notes=64, seed_window=None):
        digits = self.PI_DIGITS[:n_notes]
        return [self.digit_to_pitch[int(d)] for d in digits]


# ------------------------------------------------------------------ Tier 4 --
class MLPComposer:
    """Neural next-note predictor.

    Design choices worth calling out in the presentation:
    - CLASSIFIER, not regressor: pitches are categorical. Predicting 'note
      63.7' isn't musical; predicting a probability distribution over
      pitches and sampling from it is.
    - Top-k sampling (not argmax) at generation time. argmax gets stuck in
      loops — it will just find the most-likely note and repeat it forever.
    """
    name = "mlp"

    def __init__(self, window_size=4, hidden=(64, 64), max_iter=400,
                 top_k=3, temperature=1.0, seed=42):
        self.window_size = window_size
        self.top_k = top_k
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden,
            max_iter=max_iter,
            random_state=seed,
        )
        self._trained = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self._trained = True
        return self

    def _sample_next(self, window):
        """Sample the next pitch using top-k + temperature on the classifier's
        predicted probabilities."""
        probs = self.model.predict_proba(np.asarray(window).reshape(1, -1))[0]
        classes = self.model.classes_

        # Temperature sharpens (<1) or flattens (>1) the distribution
        if self.temperature != 1.0:
            logp = np.log(probs + 1e-12) / self.temperature
            probs = np.exp(logp - logp.max())
            probs = probs / probs.sum()

        # Keep only the top-k most likely classes
        k = min(self.top_k, len(probs))
        top_idx = np.argpartition(probs, -k)[-k:]
        top_probs = probs[top_idx]
        top_probs = top_probs / top_probs.sum()
        chosen = self.rng.choice(top_idx, p=top_probs)
        return int(classes[chosen])

    def generate(self, n_notes=64, seed_window=None):
        if not self._trained:
            raise RuntimeError("Call .fit(X, y) before .generate(...)")
        if seed_window is None:
            # Default seed: a simple ascending fragment of C major
            seed_window = [60, 62, 64, 65][: self.window_size]
        window = [int(p) for p in seed_window[-self.window_size :]]
        out = list(window)
        for _ in range(n_notes - len(window)):
            nxt = self._sample_next(window)
            out.append(nxt)
            window = window[1:] + [nxt]
        return [int(p) for p in out]


# ------------------------------------------------------------- util: to MIDI --
def pitches_to_midi(pitches, out_path, note_length=0.3, velocity=90):
    """Write a list of pitches to a MIDI file as a simple monophonic line."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    t = 0.0
    for p in pitches:
        piano.notes.append(
            pretty_midi.Note(velocity=velocity, pitch=int(p),
                             start=t, end=t + note_length)
        )
        t += note_length
    pm.instruments.append(piano)
    pm.write(out_path)
