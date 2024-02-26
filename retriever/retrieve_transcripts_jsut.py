import glob
import pandas as pd
import os
import yaml
import argparse
# %%
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="path to preprocess.yaml")
args = parser.parse_args()

config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
corpus_path = config["path"]["corpus_path"]
corpus_name = config["dataset"]
raw_path = config["path"]["raw_path"]

transcript_files = glob.glob(corpus_path + "/*/transcript_utf8.txt")
raw_path_ = raw_path + "/" + corpus_path
# %%
if not os.path.exists(raw_path_):
    os.makedirs(raw_path_)
for transcript in transcript_files:
    with open(transcript, mode='r') as f:
        lines = f.readlines()
    for line in lines:
        filename, text = line.split(':')
        with open(raw_path_ + filename + '.lab', mode='w') as f:
            f.write(text.strip('\n'))
