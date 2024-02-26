# raw_dataにある書記素のlabをすべてlab_path/speaker_name以下に音素のlabで置きます
# その後、phoneme_align_htsを用いてアライメントを取得し、置き換えます
# phoneme_align_htsの準備ができていることを確認してください

import subprocess
from pathlib import Path
import argparse
import shutil
import sys
import yaml
import re

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to config folder")
parser.add_argument("--corpus", type=str, required=True, help="name of corpus")
parser.add_argument("--speaker", type=str, required=False, default=None, help="set speaker name when to preprocess only one speaker")
args = parser.parse_args()

config = yaml.load(open(args.config + "/preprocess_" + args.corpus + ".yaml", "r"), Loader=yaml.FullLoader)
raw_path = Path(config["path"]["raw_path"])
lab_path = Path(config["path"]["lab_path"])
lang = config["text"]["language"]

if not lab_path.exists():
    lab_path.mkdir(parents=True)

for folder in raw_path.iterdir():
    if args.speaker != None:
        folder = raw_path / args.speaker
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
        p.wait()
        res = iter(p.stdout.readline, b'')
        res2 = ["sil"]
        for line in res:
            res2.extend(re.split(" +", line.decode("utf8").strip().replace("ˌ", "").replace("ˈ", "")))
            res2.append("pau")
        p.stdout.close()
        res2 = res2[:-3] + ["sil"]
        res2 = [r for r in res2 if r != "_:"]
        
        with open((lab_path_ / lab.name), mode="w") as f:
            f.write("\n".join(res2))

    cmd_md = [
        "python", "phoneme_alignment_hts/scripts/make_data.py", 
        "--work_dir", "phoneme_alignment_hts/tmp",
        lab_path_,
        raw_path / folder.name
    ]
    cmd_tr = [
        "perl", "./phoneme_alignment_hts/scripts/Training.pl",
        "./phoneme_alignment_hts/scripts/Config.pm",
        "phoneme_alignment_hts/src/htk",
        "phoneme_alignment_hts/tmp",
        "phoneme_alignment_hts/tmp/data/scp/train.scp",
        "phoneme_alignment_hts/tmp/data/lists/mono.list",
        "phoneme_alignment_hts/tmp/data/labels/mono.mlf"
    ]
    p = subprocess.Popen(cmd_md)
    p.wait()
    p = subprocess.Popen(cmd_tr)
    p.wait()

    shutil.rmtree(lab_path_)
    shutil.move("phoneme_alignment_hts/tmp", lab_path_)
    
    if args.speaker != None:
        exit()
