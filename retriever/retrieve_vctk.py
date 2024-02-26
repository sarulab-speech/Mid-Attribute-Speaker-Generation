import shutil
from pathlib import Path
import yaml
import argparse
import json
# %%
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="path to config folder")
args = parser.parse_args()

config = yaml.load(open(args.config + "/preprocess_VCTK.yaml", "r"), Loader=yaml.FullLoader)
corpus_path = Path(config["path"]["corpus_path"])
corpus_name = Path(config["dataset"])
preprocessed_path = Path(config["path"]["preprocessed_path"])
raw_path = Path(config["path"]["raw_path"])

speakers = {}
with open(corpus_path / "speaker-info.txt", mode="r") as f:
    lines = f.readlines()
for i, line in enumerate(lines[1:]):
    speaker_id, _, gen, _, *_ = line.split("  ")
    speakers["p" + speaker_id] = [i, gen, "en"]

with open(preprocessed_path / "speakers.json", mode="w") as f:
    json.dump(speakers, f)

gomi_list = []
with open(corpus_path.with_name("gomi_wav.list"), mode="r") as f:
    lines = f.readlines()
for line in lines[3:]:
    _, gomi = line.split(" ")
    gomi_list.append(gomi.strip().replace(".wav", ""))
print("gomi: ", gomi_list)
# %%


for speaker in speakers:
    if speaker == "p315":
        continue
    text_path = corpus_path / "txt" / speaker
    wav_path = corpus_path / "wav48" / speaker
    raw_path_ = raw_path / speaker
    if not raw_path_.exists():
       raw_path_.mkdir(parents=True)
    for text in text_path.iterdir():
        filename = text.stem
        if filename in gomi_list:
            continue
        with open(text, mode='r') as f:
            line = f.readline()
        with open((raw_path_ / filename).with_suffix(".lab"), mode='w') as f:
                f.write(line.strip('\n'))
        shutil.copyfile((wav_path / filename).with_suffix(".wav"), (raw_path_ / filename).with_suffix(".wav"))
