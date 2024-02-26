import os
import json

import torch
import torch.nn as nn

class Lang_Discriminator(nn.Module):
    """language discriminator"""

    def __init__(self, preprocess_config, model_config):
        super(Lang_Discriminator, self).__init__()
        self.model_config = model_config
        self.hidden_dim = model_config["discriminator"]["hidden"]

        self.discriminator = nn.LSTM(
            preprocess_config["mel"]["n_mel_channels"],
            model_config["discriminator"]["hidden"]
        )
        self.linear = nn.Linear(
            model_config["discriminator"]["hidden"],
            len(preprocess_config["speaker_generation"]["metadata"]["language"])
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, mels):
        _, output = self.discriminator(mels)
        output = self.linear(output[0].view(-1, self.hidden_dim))
        output = self.softmax(output)

        return output