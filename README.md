# Digital Composer — CS35 Final Project

A tiered music-generation system. Each tier is more "intelligent" than the
last, and we measure exactly how much that intelligence buys us.

**Author:** Jiselle Nguyen
**Course:** CS35, Spring 2026


## Tiers

| Tier | Composer           | What it does                                        |
|------|--------------------|-----------------------------------------------------|
| 1    | RandomComposer     | Uniform random pitch across the piano range         |
| 2    | ScaleRandomComposer| Uniform random pitch, restricted to C major         |
| 3    | PiComposer         | Digits of π mapped deterministically to scale degrees|
| 4    | MLPComposer        | MLPClassifier trained to predict next note from context|


## What's in this folder
- `composers.py` — the four generator tiers
- `flatten_midis.py` — MIDI → CSV preprocessor
- `train_and_generate.py` — trains the MLP and saves the model
- `snippet_generator.py` — the parameterized tool (main deliverable)
- `output/model.pkl` — the pre-trained model
- `output/snippets/` — five demo MIDI files


## Data

This project trains on the NES Music Database (NES-MDB), available at:
https://github.com/chrisdonahue/nesmdb

Download the MIDI format archive, extract it, and place the MIDI files in
`data/nesmdb_midi/`. Then run:

    python3 flatten_midis.py --midi_dir data/nesmdb_midi/train --out_csv data/nes_windows.csv
    python3 train_and_generate.py --csv data/nes_windows.csv



## Quickstart

```bash
pip install pretty_midi scikit-learn numpy pandas

python3 make_sample_midis.py       # toy data; skip if using real dataset
python3 flatten_midis.py           # flatten MIDIs -> data/windows.csv
python3 train_and_generate.py      # train MLP, evaluate, write 4 MIDIs
```

Open the `.mid` files in MuseScore, GarageBand, or any DAW to listen.


    
## Using the real NES dataset

1. Download from https://github.com/chrisdonahue/nesmdb (the "MIDI" variant)
2. Unzip into `data/nes_midis/`
3. Run:
   ```bash
   python3 flatten_midis.py --midi_dir data/nes_midis --out_csv data/nes_windows.csv
   python3 train_and_generate.py --csv data/nes_windows.csv
   ```

## Results on sample data (40 random-walk melodies, 3,040 windows)

| Composer       | Top-1 acc | Top-3 acc |
|----------------|----------:|----------:|
| random         |      3.6% |         — |
| scale_random   |      5.4% |         — |
| pi             |      5.2% |         — |
| **mlp**        | **11.0%** | **21.4%** |

The MLP is ~2–3× better than baselines on weak random-walk data. On real
melodic data (NES / classical) the gap is typically much larger.

## Design choices worth mentioning in the presentation

- **Classifier, not regressor** — pitches are categorical. MLPClassifier
  gives us a probability distribution to sample from, which is how we get
  controllable creativity.
- **Top-k sampling + temperature** during generation — argmax alone gets
  stuck in loops (always picks the one most-likely note).
- **Transpose everything to C** in the flattener — the model only has to
  learn one key's worth of patterns.
- **Monophonic only** — we take the melody line (instrument with most
  notes) and drop simultaneous duplicates. Polyphony is a clear stretch goal.

## What's next

- [ ] Predict duration as well as pitch (richer feature vector per window)
- [ ] Predict *intervals* instead of absolute pitches (key-independent)
- [ ] Larger window size (8–16 instead of 4) to capture phrase-level patterns
- [ ] Human listening test: 4 clips, rate 1–5 for "musicality"

## Overall Final Reflection

The project began with the idea of building a "digital composer", in which the system could take a short musical phrase and continue it by guessing the next best note, following the same logic of how AI uses prediction to craft responses. This original framing leaned heavily on the cross-section between digit recognition and musical sequences, which was very similar to work created in class homeworks before. The first plan was to train on the Lakh MIDI Dataset and generate continuations of user-provided melodies. Though, after meeting to discuss this, it was brought to my attention that this database was far too large to create a tangible project. Instead, we shifted to a much smaller NES-MDB dataset, which was stylistically consistent and added a tiered baseline structure (random -> sscale-random -> pi-music -> MLP) so that we could measure the model's value rather than just hope the notes sounded good together. We pivoted further from this system that generated music on its own into a controllable tool in which the user could outline specific parameters. By the time of our presentation, we had a project that revolved more about the user than the model itself, which is what we were aiming for!

The Final system has 5 main components: a sample MIDI generator for testing, a MIDI flattener for processing raw MIDI files into a traiing CSV, four composer classes (Random, ScaleRandom, Pi, MLP), a training script that saved the trained model to disk, and a SnippetGenerator class that wraps the trained model with 8 user-controllable parameters. These parameters are key, mode, tempo, variability, voices, dynamics, note density, and duration. The snippet generator can produce 10-second MIDI files in any key, in any of five modes (major, minor, dorian, mixolydian, pentatonic) in monophonic or polyphonic textures, with controllable rhythm density and dynamic envelopes (ie. crescendo and diminuendo). Additionally, rerunning with different parameters does not require retraining, as everything is saved in the disk.

Something that I would add if I were to extend the project I would consider adding a duration prediction alongside pitch prediction. Currently, the rhythm is generated procedually from the density parameter. Possibly training the MLP to learn how rhythm and melody interact would make a richer and more musically interesting system. Maybe another thing I would include would be a language-based input system, like giving the snippet generator a creative prompt instead of telling it exactly which key, mode, or tempo to produce music in. For example, one could write "make me a really sad funeral piece in A minor" or "please make a dramatic introduction for a supervillain."

Lastly, something I want to highlight is that although I had a partner in this project, my partner did not contribute to any of the aforementioned pieces of the Digital Composer. Throughout the project, I consistently found datasets, built parts of the composer, trouble-shooted and experimented with different kinds of samples, and improved the model to its completed state. I would say my partner contributed little to nothing in the entierty of this project, including in the slides and during the presentation itself. I chose to write and submit this reflection by myself because that is what I felt was most appropriate, as I wanted to write in detail about my process of working through this project. I hope that this perspective is understood and I wish that me and my partner will be graded accordingly to the work that we each contributed.