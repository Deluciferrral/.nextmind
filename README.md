# NextMind lightweight Python reader and shielding utilities

This small toolkit provides a defensive reader for NextMind `.raw` and `.inf` files
and a set of simple "shielding" functions (filters and artifact rejection).

Files added:
- `nextmind_reader.py` — helper functions to discover recordings and load `.raw`/`.inf` files.
- `shielding.py` — bandpass/notch filters and a simple threshold-based artifact rejection.
- `run_example.py` — example script to load a recording, apply shielding, and save cleaned data.
- `requirements.txt` — `numpy` and `scipy` required.

Quickstart
1. Create a virtualenv and install requirements:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Run the example on `recording/0` (from the `.nextmind` folder):

```powershell
python run_example.py recording\0
```

Notes and assumptions
- The reader uses heuristics to infer data dtype and channel count when metadata isn't available.
- "Shielding" here means simple preprocessing (bandpass + notch + threshold-based artifact removal).
- If you have richer metadata (sample rate, channel names, units), pass those to the functions or extend `read_inf`.

Next steps
- If you want fully-featured EEG processing, I can adapt this to use `mne` and add channel montages, ICA-based artifact removal, and better event parsing.
