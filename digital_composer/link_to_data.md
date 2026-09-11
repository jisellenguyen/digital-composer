## Data

This project trains on the NES Music Database (NES-MDB), available at:
https://github.com/chrisdonahue/nesmdb

Download the MIDI format archive, extract it, and place the MIDI files in
`data/nesmdb_midi/`. Then run:

    python3 flatten_midis.py --midi_dir data/nesmdb_midi/train --out_csv data/nes_windows.csv
    python3 train_and_generate.py --csv data/nes_windows.csv