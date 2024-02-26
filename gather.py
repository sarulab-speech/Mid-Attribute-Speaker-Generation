from pathlib import Path
import argparse
import json
import yaml

from sklearn.preprocessing import StandardScaler
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to config folder")
args = parser.parse_args()

config_path = Path(args.config)
preprocess_path = config_path / "preprocess.yaml"
preprocess = yaml.load(open(preprocess_path, "r"), Loader=yaml.FullLoader)
normalize_energy = preprocess["energy"]["normalization"]
normalize_pitch = preprocess["pitch"]["normalization"]
speakers = {}
stats = {"pitch": [float("inf"), -float("inf")], "energy": [float("inf"), -float("inf")]}
pitch_scaler = StandardScaler()
energy_scaler = StandardScaler()
total = 0
for preprocess in config_path.glob("preprocess_*.yaml"):
    config = yaml.load(open(preprocess, "r"), Loader=yaml.FullLoader)
    speaker_json = json.load(open(Path(config["path"]["preprocessed_path"]) / "speakers.json", "r"))
    energy_path = Path(config["path"]["preprocessed_path"]) / "energy"
    pitch_path = Path(config["path"]["preprocessed_path"]) / "pitch"
    for spk in speaker_json:
        speakers[spk] = [total, *speaker_json[spk][1:]]
        total += 1
    for energy_npy in energy_path.iterdir():
        energy = np.load(energy_npy)
        energy_scaler.partial_fit(energy.reshape((-1, 1)))
        stats["energy"][1] = max(stats["energy"][1], max(energy))
        stats["energy"][0] = min(stats["energy"][0], min(energy))
    for pitch_npy in pitch_path.iterdir():
        pitch = np.load(pitch_npy)
        pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
        stats["pitch"][1] = max(stats["pitch"][1], max(pitch))
        stats["pitch"][0] = min(stats["pitch"][0], min(pitch))

if normalize_pitch:
    pitch_mean = pitch_scaler.mean_[0]
    pitch_std = pitch_scaler.scale_[0]
else:
    pitch_mean = 0
    pitch_std = 1

if normalize_energy:
    energy_mean = energy_scaler.mean_[0]
    energy_std = energy_scaler.scale_[0]
else:
    energy_mean = 0
    energy_std = 1

stats = {
    "pitch": [
            float((stats["pitch"][0] - pitch_mean) / pitch_std),
            float((stats["pitch"][1] - pitch_mean) / pitch_std),
            float(pitch_mean),
            float(pitch_std)
        ],
        "energy": [
            float((stats["energy"][0] - energy_mean) / energy_std),
            float((stats["energy"][1] - energy_mean) / energy_std),
            float(energy_mean),
            float(energy_std),
        ],
    }

with open(config_path / "speakers.json", "w") as f:
    json.dump(speakers, f)

with open(config_path / "stats.json", "w") as f:
    json.dump(stats, f)