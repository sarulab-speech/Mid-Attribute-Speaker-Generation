#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
import sys
import glob
import os
import librosa
import numpy as np
import dataset
import random
import tqdm
import soundfile as sf
from scipy.io import wavfile
from hparam import hparam as hp
from os.path import expanduser, join, exists
from multitask import pool_map
from functools import partial
from data_function import TextMelLoaderStatic

# downloaded dataset path
#audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))                                        
data_template = '{dataset}_{spkr}_{gender}_{language}.npy'

def collect_datasets():
    print('collecting datasets')
    datasets = []
    for name, kwarg in hp.datasets.items():
        if name not in hp.train_datasets:
            continue
        kwarg['root'] = expanduser(kwarg['root'])
        print('collect {}'.format(name))
        ds = getattr(dataset, name, None)(**kwarg)
        datasets.append(ds)
    print('totally {} dataset'.format(len(datasets)))
    return datasets

datasets = collect_datasets()

def process_wav(utter_path, utter_min_len, mel_calculator):
    # get log-mel spectrogram from utterance

    #utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio
    utter, sr = sf.read(utter_path)
    if sr != hp.data.sr:
        utter = librosa.resample(utter, sr, hp.data.sr)
    intervals = librosa.effects.split(utter, top_db=100)         # voice activity detection 
    # for vctk dataset use top_db=100
    if len(intervals) == 0:
        print('Warning: wav: {} has no interval, try to improve top_db'.format(utter_path))
    
    specs = []
    for interval in intervals:
        if (interval[1]-interval[0]) < utter_min_len:           # If partial utterance is too short
            print(interval[0], interval[1], utter_min_len)
            continue
        utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
        try:
            S = mel_calculator.get_mel(utter_part)
        except Exception as e:
            print(e)
            print('error for {}'.format(utter_path))
            continue
        i = 0
        while S.shape[1] >= hp.data.tisv_frame * (i+1):
            specs.append(S[:, int(hp.data.tisv_frame*i):int(hp.data.tisv_frame*(i+1))])
            i += 0.5
    return specs
        
def process_spkr_wavlist(data, ds, utter_min_len, test_spkrlist):
    mel_calculator = TextMelLoaderStatic(hp)
    for spkr, wavlist in data:
        specs = []
        gender = ds.get_gender(spkr)
        path = data_template.format(dataset=ds._name, spkr=spkr, gender=gender, language=ds._language)
        if spkr in test_spkrlist:
            path = join(hp.data.test_path, path)
        else:
            path = join(hp.data.train_path, path)
        if exists(path):
            continue
        for wav in wavlist:
            buf = process_wav(wav, utter_min_len, mel_calculator)
            if len(buf) == 0:
                print('Warning: spkr {}\'s wav: {} has no specs, try to improve top_db'.format(spkr, wav))
            specs.extend(buf)
        specs = np.array(specs)
        # print(specs.shape)
        np.save(path, specs) # save the spec

def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
    """
    print("start text independent utterance feature extraction")
    os.makedirs(hp.data.train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(hp.data.test_path, exist_ok=True)    # make folder to save test file

    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window)   # lower bound of utterance length
    
    total_speaker_num = sum(d.spkr_number for d in datasets)
    train_speaker_num = total_speaker_num - len(datasets) * hp.test_speaker_per_dataset
    
    print("total speaker number : %d"%total_speaker_num)
    print("train : %d, test : %d, test_speaker_per_dataset: %d" % \
           (train_speaker_num, total_speaker_num-train_speaker_num, hp.test_speaker_per_dataset))
    speaker_number = 0
    for ds in datasets:
        print('process {}, lang: {}, total speaker: {}'.format(ds._name, ds._language, ds.spkr_number))
        spkrlist = list(ds.spkrs.keys())
        #test_spkrlist = random.sample(spkrlist, hp.test_speaker_per_dataset)
        test_spkrlist = ds.test_speakers
        print('test spkrs: {}'.format(','.join(test_spkrlist)))
        data = [(s, w) for s, w in ds.spkr_wavlist()]
        process_funct = partial(process_spkr_wavlist, ds=ds, utter_min_len=utter_min_len, test_spkrlist=test_spkrlist)
        _ = pool_map(process_funct, data, mode='Process', max_workers=16)

if __name__ == "__main__":
    save_spectrogram_tisv()
