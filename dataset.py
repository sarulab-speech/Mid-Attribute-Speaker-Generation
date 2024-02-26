import json
import math
import os

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset

from text import symbols, text_to_sequence
from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.use_accent = preprocess_config["preprocessing"]["accent"]["use_accent"]
        self.accent_to_id = {'0':0, '[':1, ']':2, '#':3}

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        
        self.speaker_meta = preprocess_config["preprocessing"]["speaker_generation"]["metadata"]
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker][0]
        speaker_meta = {}
        for i, meta in enumerate(self.speaker_meta):
            speaker_meta[meta] = self.speaker_map[speaker][i+1]
        raw_text = self.raw_text[idx]
        phone = np.array([self.symbol_to_id[t] for t in self.text[idx].replace("{", "").replace("}", "").split()])
        if self.use_accent:
            with open(os.path.join(self.preprocessed_path, "accent",basename+ '.accent')) as f:
                accent = f.read()
            accent = [self.accent_to_id[t] for t in accent]
            accent = np.array(accent[:len(phone)])
        else:
            accent = np.array([4] * len(phone))

        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "speaker_name": speaker,
            "speaker_meta": speaker_meta,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "accent": accent
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        accents = [data[idx]["accent"] for idx in idxs]

        # turn speaker_meta to one-hot vector
        speaker_meta = list(
            np.concatenate([np.eye(len(self.speaker_meta[meta]))[self.speaker_meta[meta][val]] for meta, val in data[idx]["speaker_meta"].items()]) for idx in idxs
        )

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        speaker_meta = np.array(speaker_meta)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        
        if self.use_accent:
            accents = pad_1D(accents)
            return (
                ids,
                raw_texts,
                speakers,
                texts,
                text_lens,
                max(text_lens),
                mels,
                mel_lens,
                max(mel_lens),
                pitches,
                energies,
                durations,
                speaker_meta,
                accents
            )
        else:
            return (
                ids,
                raw_texts,
                speakers,
                texts,
                text_lens,
                max(text_lens),
                mels,
                mel_lens,
                max(mel_lens),
                pitches,
                energies,
                durations,
                speaker_meta
            )


    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class ConcatDataset(ConcatDataset):
    def __init__(self, config, datasets) -> None:
        super(ConcatDataset, self).__init__(datasets)
        self.collate_fn = datasets[0].collate_fn
        with open(os.path.join(config, "stats.json")) as f:
            self.stats = json.load(f)
        with open(os.path.join(config, "speakers.json")) as f:
            self.speaker_map = json.load(f)
    
    def __getitem__(self, idx):
        sample = super(ConcatDataset, self).__getitem__(idx)
        sample["pitch"] = (sample["pitch"] - self.stats["pitch"][2]) / self.stats["pitch"][3]
        sample["energy"] = (sample["energy"] - self.stats["energy"][2]) / self.stats["energy"][3]
        sample["speaker"] = self.speaker_map[sample["speaker_name"]][0]
        return sample
    

class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

        self.use_accent = preprocess_config["preprocessing"]["accent"]["use_accent"]
        self.accent_to_id = {'0':0, '[':1, ']':2, '#':3}

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        accent = None
        if self.use_accent:
            with open(os.path.join(self.preprocessed_path, "accent",basename+ '.accent')) as f:
                accent = f.read()
            accent = [self.accent_to_id[t] for t in accent]
            accent = np.array(accent[:len(phone)])

        return (basename, speaker_id, phone, raw_text,accent)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])
        if self.use_accent:
            accents = [d[4] for d in data]

        texts = pad_1D(texts)
        accents = pad_1D(accents)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens), accents


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
