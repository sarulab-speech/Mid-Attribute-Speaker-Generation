#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import numpy as np
import os
import random
from random import shuffle
import torch
from torch.utils.data import Dataset
from os.path import join, basename

from hparam import hparam as hp
from utils import mfccs_and_spec, count_label

class SpeakerDatasetTIMIT(Dataset):
    
    def __init__(self):

        if hp.training:
            self.path = hp.data.train_path_unprocessed
            self.utterance_number = hp.train.M
        else:
            self.path = hp.data.test_path_unprocessed
            self.utterance_number = hp.test.M
        self.speakers = glob.glob(os.path.dirname(self.path))
        shuffle(self.speakers)
        
    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        
        speaker = self.speakers[idx]
        wav_files = glob.glob(speaker+'/*.WAV')
        shuffle(wav_files)
        wav_files = wav_files[0:self.utterance_number]
        
        mel_dbs = []
        for f in wav_files:
            _, mel_db, _ = mfccs_and_spec(f, wav_process = True)
            mel_dbs.append(mel_db)
        return torch.Tensor(mel_dbs)

class SpeakerDataset(Dataset):

    def __init__(self, shuffle=True, utter_start=0):
        # data path
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
        self.da_on = hp.model.da_on
        self.datasets = hp.train_datasets
        self.file_list = self.collect_filelist()
        self.shuffle=shuffle
        self.utter_start = utter_start
        self.lang2filelist = self.groupby(self.da_on)
        self.langs = list(self.lang2filelist)
        self.langs.sort()
        self.file2data = {}
        for f in self.file_list:
            self.file2data[f] = np.load(join(self.path, f))

    def collect_filelist(self):
        filelist = []
        for d in self.datasets:
            path = join(self.path, '{}*.npy'.format(d.lower()))
            buf = [basename(item) for item in glob.glob(path)]
            filelist.extend(buf)
        return filelist

    def groupby(self, da_on):
        lang2filelist = {}
        for f in self.file_list:
            decoded = self.decode_filename(f)
            l = decoded[da_on]
            if l not in lang2filelist:
                lang2filelist[l] = []
            lang2filelist[l].append(f)
        return lang2filelist
    
    @staticmethod
    def decode_filename(filename):
        d, s, g, l = basename(filename)[:-4].split('_')
        return {'dataset': d, 'spkr': s, 'gender': g, 'language'
        :l}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.shuffle:
            lang = idx % len(self.langs)
            temp = self.lang2filelist[self.langs[lang]]
            selected_file = random.sample(temp, 1)[0]  # select random speaker
        else:
            selected_file = self.file_list[idx]
            lang = self.langs.index(self.decode_filename(selected_file)[self.da_on])
        utters = self.file2data[selected_file]
        valid_indices = list(range(utters.shape[0]))
        if self.utter_num > len(valid_indices):
            utter_index = random.choices(valid_indices, k=self.utter_num)
        else:
            utter_index = random.sample(valid_indices, self.utter_num)   # select M utterances per speaker
        utterance = utters[utter_index]       
        utterance = utterance[:,:,:hp.data.tisv_frame]               # max length here

        utterance = torch.FloatTensor(np.transpose(utterance, axes=(0,2,1)))     # transpose [batch, frames, n_mels]
        lang = torch.LongTensor([lang])
        return utterance, lang

class Collate:
    def __init__(self, variable_length=True):
        self.variable_length = variable_length
        self.lang_type = torch.FloatTensor if count_label(hp) == 1 else torch.LongTensor
        self.lower = int(hp.data.tisv_frame - 0.4 / hp.data.hop * hp.data.sr)

    def __call__(self, batch):
        mels, langs = list(zip(*batch))
        if self.variable_length:
            length = random.randint(self.lower, hp.data.tisv_frame)
            new_mels = []
            for mel in mels:
                p = random.randint(0, mel.shape[1] - length)
                new_mel = mel[:, p:p+length, :]
                new_mels.append(new_mel)
            new_mels = torch.stack(new_mels)
        else:
            new_mels = torch.stack(mels)
        langs = torch.stack(langs).\
                repeat(1, new_mels.shape[1]).\
                flatten().type(self.lang_type)
        new_batch = {'mels': new_mels, 'langs': langs}
        return new_batch


