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

import random
import numpy as np
import torch
import torch.utils.data
from time import time
from os.path import join

import common.layers as layers
from common.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu
from os.path import basename

class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        if batch[0][1] is not None:
            # Right zero-pad mel-spec
            num_mels = batch[0][1].size(0)
            max_target_len = max([x[1].size(1) for x in batch])
            if max_target_len % self.n_frames_per_step != 0:
                max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
                assert max_target_len % self.n_frames_per_step == 0

            # include mel padded and gate padded
            mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
            mel_padded.zero_()
            gate_padded = torch.FloatTensor(len(batch), max_target_len)
            gate_padded.zero_()
            output_lengths = torch.LongTensor(len(batch))
            for i in range(len(ids_sorted_decreasing)):
                mel = batch[ids_sorted_decreasing[i]][1]
                mel_padded[i, :, :mel.size(1)] = mel
                gate_padded[i, mel.size(1)-1:] = 1
                output_lengths[i] = mel.size(1)
        else:
            mel_padded = gate_padded = output_lengths = None
        
        # dvec
        dvecs = []
        for i in range(len(ids_sorted_decreasing)):
            dvec = batch[ids_sorted_decreasing[i]][2]
            dvecs.append(dvec)
        dvecs = torch.stack(dvecs)

        # count number of items - characters in text
        len_x = [x[3] for x in batch]
        len_x = torch.Tensor(len_x)
        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, len_x, dvecs

def batch_to_gpu(batch):
    text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths, len_x, dvecs = batch
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    max_len = torch.max(input_lengths.data).item()
    if mel_padded is not None:
        mel_padded = to_gpu(mel_padded).float()
    if gate_padded is not None:
        gate_padded = to_gpu(gate_padded).float()
    if output_lengths is not None:
        output_lengths = to_gpu(output_lengths).long()
        len_x = torch.sum(output_lengths)
    else:
        len_x = None
    dvecs = to_gpu(dvecs).float()
    x = (text_padded, input_lengths, mel_padded, max_len, output_lengths, dvecs)
    y = (mel_padded, gate_padded)
    return (x, y, len_x)

class TextMelLoaderStatic(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, args, stft=None, sr=16000):
        if stft is None:
            self.sampling_rate = args.data.sr
            # self.stft = layers.TacotronSTFT(
            #     args.data.nfft, int(args.data.hop * args.data.sr), int(args.data.window * args.data.sr),
            #     args.data.nmels, args.data.sr, args.data.fmin,
            #     args.data.fmax)
            self.stft = layers.TacotronSTFT(
                args.data.nfft, args.data.hop, args.data.window, # directly use
                args.data.nmels, args.data.sr, args.data.fmin,
                args.data.fmax)
        else:
            self.sampling_rate = sr
            self.stft = stft

    @classmethod
    def fromaudioargs(TextMelLoaderStatic, nfft, hop, window, nmels, sr, fmin, fmax):
        stft = layers.TacotronSTFT(nfft, hop, window, nmels, sr, fmin, fmax)
        return TextMelLoaderStatic(None, stft, sr)
    
    @classmethod
    def fromtacoargs(TextMelLoaderStatic, args):
        stft = layers.TacotronSTFT(args.filter_length, args.hop_length, args.win_length, args.n_mel_channels, args.sampling_rate, args.mel_fmin, args.mel_fmax)
        sr = args.sampling_rate
        return TextMelLoaderStatic(None, stft, sr)

    def get_mel(self, audio):
        audio = torch.FloatTensor(audio.astype(np.float32))
        #audio_norm = audio / np.abs(audio).max()
        audio_norm = audio.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        melspec = melspec.numpy()
        return melspec
