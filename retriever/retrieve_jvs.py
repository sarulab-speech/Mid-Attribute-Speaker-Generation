import shutil
import pandas as pd
import os
import yaml
import argparse
# %%
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="path to config folder")
args = parser.parse_args()

config = yaml.load(open(args.config + "/preprocess_JVS.yaml", "r"), Loader=yaml.FullLoader)
corpus_path = config["path"]["corpus_path"]
corpus_name = config["dataset"]
raw_path = config["path"]["raw_path"]

raw_path_ = raw_path + "/" + corpus_path
# %%

for i in range(1, 101):
    speaker_path = corpus_path + "/jvs" + "{:0>3d}/".format(i)
    target_fordars = ["parallel100/", "nonpara30/"]
    raw_path_ = raw_path + "/jvs" + "{:0>3d}/".format(i) 
    if not os.path.exists(raw_path_):
        os.makedirs(raw_path_)
    for target in target_fordars:
        transcript = speaker_path + target + "transcripts_utf8.txt"
        with open(transcript, mode='r') as f:
            lines = f.readlines()
        for line in lines:
            filename, text = line.split(':')
            if os.path.exists(speaker_path + target + "wav24kHz16bit/" + filename + ".wav"):
                with open(raw_path_ + filename + '.lab', mode='w') as f:
                    f.write(text.strip('\n'))
                shutil.copyfile(speaker_path + target + "wav24kHz16bit/" + filename + ".wav", raw_path_ + filename + ".wav")
