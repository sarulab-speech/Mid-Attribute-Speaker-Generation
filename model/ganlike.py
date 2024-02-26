import torch
import torch.nn as nn
import random
import copy

from model import FastSpeech2
_temp = __import__("Multilingual-Speaker-Encoder-with-Domain-Adaptation", globals(), locals(), ['SpeechEmbedder', 'GE2ELoss'], 0)
SpeechEmbedder = _temp.SpeechEmbedder

class GANLike(nn.Module):

    def __init__(self, model: FastSpeech2, discriminator: SpeechEmbedder, train_config):
        super(GANLike, self).__init__()
        self.model = model
        self.discriminator = discriminator
        self.batch_size = train_config["optimizer"]["batch_size"]
    
    def forward(self, device, batch):
        # monolingual TTS train
        # Forward
        accents = batch[-1]
        speaker_meta = batch[-2]
        batch = batch[:-2]
        output = self.model(*(batch[2:]),accents=accents,speaker_meta=speaker_meta)

        # Crosslingual TTS train
        # Shuffle speaker to make cross-lingual case
        reorder = random.sample(list(range(self.batch_size)), self.batch_size)
        speakers = torch.stack([batch[2][reorder[i]] for i in range(self.batch_size)])
        speaker_meta_original = copy.deepcopy(speaker_meta)
        speaker_meta = torch.stack([speaker_meta[reorder[i]] for i in range(self.batch_size)])
        _batch = batch[:2] + tuple([speakers]) + batch[3:]
        # Forward
        _output = self.model(*(_batch[2:]),accents=accents,speaker_meta=speaker_meta)
        # Reshape batch
        # output[0]: predicted mel(batch N, max_len_mel, n_mels)  output[7]: mel mask
        # speaker_meta[2]: ==japanese speaker
        max_len = _output[0].shape[1]
        max_len_r = max_len // 150 + 1
        n_mels = _output[0].shape[2]
        batch_r_m = torch.cat([_output[0], torch.zeros(self.batch_size, max_len_r*150-max_len, n_mels).to(device)], dim=1).view(self.batch_size * max_len_r, 150, n_mels)
        langs = speaker_meta[:,2].view(-1, 1).repeat(1, max_len_r).view(-1)
        # Forward
        output_r = self.discriminator(batch_r_m)

        return output, langs, output_r, max_len_r