# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import numpy as np
from scipy.io.wavfile import read
import torch
import os
import librosa
import argparse
import json
import torch.nn.functional as F

def extract_spkr_from_filename(filename):
    if '_' in filename:
        return filename.split('_')[0]
    else:
        return filename.split('.')[0]

class ParseFromConfigFile(argparse.Action):

    def __init__(self, option_strings, type, dest, help=None, required=False):
        super(ParseFromConfigFile, self).__init__(option_strings=option_strings, type=type, dest=dest, help=help, required=required)

    def __call__(self, parser, namespace, values, option_string):
        with open(values, 'r') as f:
            data = json.load(f)

        for group in data.keys():
            for k,v in data[group].items():
                underscore_k = k.replace('-', '_')
                setattr(namespace, underscore_k, v)

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).byte()
    mask = torch.le(mask, 0)
    return mask

def mu_law(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return fx

def encode_mu_law(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)

def decode_mu_law(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x

def load_wav_to_torch(full_path, sr=16000):
    #data, sampling_rate = librosa.load(full_path, sr)
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(dataset_path, filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        def split_line(root, line):
            parts = line.strip().split(split)
            if len(parts) > 2:
                raise Exception(
                    "incorrect line format for file: {}".format(filename))
            path = os.path.join(root, parts[0])
            text = parts[1]
            return path,text
        filepaths_and_text = [split_line(dataset_path, line) for line in f]
    return filepaths_and_text


def to_gpu(x):
    if x is None:
        return x
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x

def std_dataset(name):
    if name.lower() == 'librispeech':
        out = 'LibriSpeech'
    else:
        out = name.upper()
    return out
    
def normalize_wav(wav):
    if wav.dtype == np.int16:
        wav = wav.astype(np.int32)
    wav = wav / np.abs(wav).max()                         # normalize the utter 
    return wav

def segment_mel(mel, dim, segment_length=130):
    '''
        @Overview
            segment mel-spec to fixed length
        @Params
            mel: mel-spec
            dim: time dimension of mel
            segment_length: length of each segment
        @Returns
            segmented_mel
    '''
    buf = []
    assert mel.dim() == 2
    # [#nmels, T]
    if dim == 0:
        mel = mel.T
    length = mel.shape[1]
    pad_value = 0
    if length < segment_length:
        mel = F.pad(mel, (0, segment_length - length), "constant", pad_value)
        buf.append(mel)
    else:
        i = 0
        while (i+1) * segment_length <= length:
            segment = mel[:,  i*segment_length: (i+1)*segment_length]
            segment = F.pad(segment, (0, segment_length - segment.shape[1]), "constant", pad_value)
            i += 1
            buf.append(segment)
    mel = torch.stack(buf) # [#N, #nmels, #segment_length]
    if dim == 0:
        mel = mel.transpose(1, 2)
    return mel

def read_speaker_list(path):
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    spkrs = []
    for line in lines:
        line = line.split()
        if len(line) == 1: # normal format
            spkr = line[0]
        else:              # consistency format
            spkr = line[1]
        spkrs.append(spkr)
    return spkrs
