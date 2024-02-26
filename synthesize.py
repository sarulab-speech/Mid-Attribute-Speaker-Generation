import re
import argparse
from string import punctuation
from pathlib import Path
import subprocess

import torch
import yaml
import numpy as np
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence, symbols
import pyopenjtalk
from prepare_tg_accent_jsut import pp_symbols
from convert_label import openjtalk2julius

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)

    cmd = [
        "espeak-ng",
        "--ipa", "--sep",
        "-v", "en",
        "-q",
        '"{}"'.format(text)
    ]
    p = subprocess.Popen(cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    res = iter(p.stdout.readline, b'')
    res2 = []
    for line in res:
        res2.extend(re.split(" +", line.decode("utf8").strip().replace("ˌ", "").replace("ˈ", "")))
        res2.append("pau")
    p.stdout.close()
    res2 = [r for r in res2[:-1] if r != "_:"]
    phones = [openjtalk2julius(r) for r in res2]

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def preprocess_japanese(text:str):
    fullcontext_labels = pyopenjtalk.extract_fullcontext(text)
    phonemes , accents = pp_symbols(fullcontext_labels)
    phonemes = [openjtalk2julius(p) for p in phonemes if p != '']
    return phonemes, accents



def synthesize(args, model, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        speaker_meta = batch[-2].view(1, -1)
        accents = batch[-1]
        batch = batch[:-2]
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
                accents=accents,
                speaker_meta=speaker_meta
            )
            synth_samples(
                args,
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--restore_step",
        type=int,
        required=True
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        required=True
    )
    parser.add_argument(
        "--use_accent",
        action="store_true"
    )
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "-s",
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to config folder",
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    # Check source texts
    assert args.text is not None

    config_path = Path(args.config)

    # Read Config
    preprocess_config = yaml.load(
        open(config_path / "preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open(config_path / "model.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open(config_path / "train.yaml", "r"), Loader=yaml.FullLoader
    )
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    accent_to_id = {'0':0, '[':1, ']':2, '#':3}

    ids = raw_texts = [args.text[:100]]
    speakers = np.array([args.speaker_id])
    speaker_meta = np.zeros(sum([len(x) for x in preprocess_config["speaker_generation"]["metadata"].values()]))
    if args.language == "en":
        texts = np.array([preprocess_english(args.text, preprocess_config)])
    elif args.language == "zh":
        texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
    elif args.language == "ja":
        phonemes, accents = preprocess_japanese(args.text)
        print(phonemes,accents)
        texts = np.array([[symbol_to_id[t] for t in phonemes]])
    if args.use_accent:
        accents = np.array([[accent_to_id[a] for a in accents]])
    else:
        accents = np.array([4]*len(texts[0]))

    text_lens = np.array([len(texts[0])])
    print(text_lens)
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens), speaker_meta, accents)]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(args, model, configs, vocoder, batchs, control_values)
