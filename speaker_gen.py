import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence, symbols
import pyopenjtalk
from prepare_tg_accent_jsut import pp_symbols
from convert_label import openjtalk2julius

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def synthesize(model, step, configs, speaker_meta):
    preprocess_config, model_config, train_config = configs

    generated_embedding = model.speaker_gen(speaker_meta)
    x = generated_embedding.to('cpu').detach().numpy().copy()
    np.save(train_config["path"]["result_path"] + "/generated_speaker", x)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--speaker_gender",
        type=str,
        default="F",
        help="speaker metadata(gender), F or M",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()


    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    speaker_meta = torch.from_numpy(
        np.eye(2)[preprocess_config["preprocessing"]["speaker_generation"]["metadata"]["gender"][args.speaker_gender]]
    ).view(1, -1).float().to(device)

    synthesize(model, args.restore_step, configs, speaker_meta)
