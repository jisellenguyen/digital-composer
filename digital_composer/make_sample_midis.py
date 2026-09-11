"""
make_sample_midis.py

Creates a small folder of sample MIDI files so the rest of the pipeline is
testable without needing to download the NES dataset yet. These are simple
diatonic melodies in C major — enough structure for the MLP to learn something
meaningful in seconds.

For the real project, replace this with the Classic NES MIDI dataset:
https://github.com/chrisdonahue/nesmdb  (or Lakh if you're feeling brave)
"""

import os
import random
import pretty_midi

random.seed(0)

OUT_DIR = "data/sample_midis"
os.makedirs(OUT_DIR, exist_ok=True)

# C major scale across two octaves (MIDI pitch numbers)
C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84]


def make_melody(filename, n_notes=40, note_length=0.3):
    """Build a short melody where each next note is within ~3 scale steps
    of the previous one — this gives the MLP a learnable local pattern."""
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # acoustic grand

    idx = random.randrange(len(C_MAJOR))
    t = 0.0
    for _ in range(n_notes):
        pitch = C_MAJOR[idx]
        piano.notes.append(
            pretty_midi.Note(velocity=90, pitch=pitch,
                             start=t, end=t + note_length)
        )
        t += note_length
        # Random-walk through the scale, bounded
        step = random.choice([-2, -1, -1, 1, 1, 2])
        idx = max(0, min(len(C_MAJOR) - 1, idx + step))

    pm.instruments.append(piano)
    pm.write(filename)


if __name__ == "__main__":
    for i in range(40):
        make_melody(os.path.join(OUT_DIR, f"sample_{i:02d}.mid"), n_notes=80)
    print(f"Wrote 40 sample melodies to {OUT_DIR}/")
