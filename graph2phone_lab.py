# raw_dataにある書記素のlabをすべてlab_path/speaker_name以下に音素のlabで置きます
# phoneme_alignment_hts対応の形式

import subprocess
from pathlib import Path
import argparse
import sys
import yaml
import re

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to config folder")
parser.add_argument("--corpus", type=str, required=True, help="name of corpus")
args = parser.parse_args()

config = yaml.load(open(args.config + "/preprocess_" + args.corpus + ".yaml", "r"), Loader=yaml.FullLoader)
raw_path = Path(config["path"]["raw_path"])
lab_path = Path(config["path"]["lab_path"])
lang = config["text"]["language"]

if not lab_path.exists():
    lab_path.mkdir(parents=True)

for folder in raw_path.iterdir():
    lab_path_ = lab_path / folder.name
    if not lab_path_.exists():
        lab_path_.mkdir()
    for lab in folder.glob("*.lab"):
        with open(lab, mode="r") as f:
            line = f.readline()
        
        cmd = [
            "espeak-ng",
            "--ipa", "--sep",
            "-v", lang,
            "-q",
            '"{}"'.format(line)
        ]
        p = subprocess.Popen(cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        res = iter(p.stdout.readline, b'')
        res2 = ["sil"]
        for line in res:
            res2.extend(re.split(" +", line.decode("utf8").strip().replace(",", "").replace("ˈ", "")))
            res2.append("pau")
        p.stdout.close()
        res2 = res2[:-3] + ["sil"]
        res2 = [r for r in res2 if r != "_:"]
        print(lab.name, res2)
        
        with open((lab_path_ / lab.name), mode="w") as f:
            f.write("\n".join(res2))