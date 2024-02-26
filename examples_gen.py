import re
import os
import argparse
from string import punctuation
from pathlib import Path
import subprocess

import torch
import yaml
import json
import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt

from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import to_device, plot_mel, expand
from text import text_to_sequence, symbols
import pyopenjtalk
from model import FastSpeech2
from prepare_tg_accent_jsut import pp_symbols
from convert_label import openjtalk2julius

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_sample_ja = "これはボイスサンプルです。"
text_sample_en = "This is a voice sample."

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


def preprocess_japanese(text:str):
    fullcontext_labels = pyopenjtalk.extract_fullcontext(text)
    phonemes , accents = pp_symbols(fullcontext_labels)
    phonemes = [openjtalk2julius(p) for p in phonemes if p != '']
    return phonemes, accents


def synth_samples(args, targets, predictions, vocoder, model_config, preprocess_config, path, basename):

    for i in range(len(predictions[0])):
        src_len = predictions[8][i].item()
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[5][i, :src_len].detach().cpu().numpy()
        if preprocess_config["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].detach().cpu().numpy()

        with open(
                os.path.join(args.config, "stats.json")
            ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(basename)))
        plt.close()

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["audio"]["sampling_rate"]
    for wav in wav_predictions:
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)



def synthesize(args, model, step, configs, vocoder, batchs, control_values, speaker_name):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = (
            batch[0],
            batch[1],
            torch.from_numpy(batch[2]).long().to(device),
            torch.from_numpy(batch[3]).to(device),
            batch[4],
            torch.from_numpy(batch[5]).float().to(device),
            torch.from_numpy(batch[6]).long().to(device)
        )
        speaker_emb = batch[-2]
        accents = batch[-1]
        batch = batch[:-2]
        with torch.no_grad():
            # Forward
            output = model.synthesize_from_speaker_emb(
                *(batch[1:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
                accents=accents,
                speaker_emb=speaker_emb
            )
            batch = (
                batch[0],
                None,
                batch[1],
                batch[2],
                batch[3],
                batch[4]
            )
            synth_samples(
                args,
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"] + "/speakers_wsample",
                speaker_name
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
    # Read Config
    config_path = Path(args.config)
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


    control_values = args.pitch_control, args.energy_control, args.duration_control
    speaker_config = json.load(open(config_path / "speakers.json", "r"))

    output_path = Path(train_config["path"]["result_path"]) / "speakers_wsample"
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # Get model
    model = get_model(args, configs, device, train=False)
    ckpt_path = os.path.join(
        train_config["path"]["ckpt_path"],
        "{}.pth.tar".format(args.restore_step),
    )
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model"])
    model.requires_grad_ = False

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    accent_to_id = {'0':0, '[':1, ']':2, '#':3}

    ids = raw_texts = [text_sample_ja[:100]]
    phonemes, accents = preprocess_japanese(text_sample_ja)
    print(phonemes,accents)
    texts = np.array([[symbol_to_id[t] for t in phonemes]])
    accents = np.array([[accent_to_id[a] for a in accents]])
    text_lens = np.array([len(texts[0])])
    print(text_lens)

    texts_en = np.array([preprocess_english(text_sample_en, preprocess_config)])
    ids_en = raw_texts_en = [text_sample_en[:100]]
    text_lens_en = np.array([len(texts_en[0])])
    accents_en = np.array([4]*len(texts_en))
    print(text_lens_en)

    with torch.no_grad():
        for speaker_name in speaker_config:
            if speaker_name=="JSUT": continue
            speaker_id = torch.Tensor([speaker_config[speaker_name][0]]).int().to(device)
            speaker_emb = model.speaker_emb(speaker_id).to('cpu').detach().numpy().copy()
            batchs = [(ids, raw_texts, texts, text_lens, max(text_lens), speaker_emb, accents)]
            synthesize(args, model, args.restore_step, configs, vocoder, batchs, control_values, speaker_name + "_ja")
            batchs = [(ids_en, raw_texts_en, texts_en, text_lens_en, max(text_lens_en), speaker_emb, accents_en)]
            synthesize(args, model, args.restore_step, configs, vocoder, batchs, control_values, speaker_name + "_en")
            np.save((output_path / speaker_name), speaker_emb)

        for language in ["ja", "en"]:
            for gender in ["M", "F"]:
                speaker_meta = torch.from_numpy(np.concatenate([
                    np.eye(2)[preprocess_config["speaker_generation"]["metadata"]["gender"][gender]],
                    np.eye(2)[preprocess_config["speaker_generation"]["metadata"]["language"][language]]
                ], 0
                )
                ).view(1, -1).float().to(device)
                for i in range(1, (len(speaker_config) - 1) // 4 + 1):
                    speaker_name = gender + "_" + language + "gen" + "{:0>3d}".format(i)
                    speaker_emb = model.speaker_gen(speaker_meta).to('cpu').detach().numpy().copy()
                    batchs = [(ids, raw_texts, texts, text_lens, max(text_lens), speaker_emb, accents)]
                    synthesize(args, model, args.restore_step, configs, vocoder, batchs, control_values, speaker_name + "_ja")
                    batchs = [(ids_en, raw_texts_en, texts_en, text_lens_en, max(text_lens_en), speaker_emb, accents_en)]
                    synthesize(args, model, args.restore_step, configs, vocoder, batchs, control_values, speaker_name + "_en")
                    np.save((output_path / speaker_name), speaker_emb)
