import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MixtureSameFamily, Independent

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths
from .jdit import JDIT


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config, config_path):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, config_path)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["mel"]["n_mel_channels"],
        )
        self.use_jdit = model_config["jdit"]["use_jdit"]
        if self.use_jdit:
            self.jdit = JDIT(model_config=model_config,preprocess_config=preprocess_config)
            

        self.postnet = PostNet()

        self.speaker_enc = None
        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    config_path, "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
            self.speaker_enc = SpeakerMetaEncoder(preprocess_config, model_config)

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        accents=None,
        speaker_meta=None,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        output = self.encoder(texts, src_masks,accents=accents)
        if self.use_jdit:
            mel_jdit, gate_outputs, alignments = self.jdit(output, mels, src_lens)

        if self.speaker_emb is not None:
            speaker_emb_s = self.speaker_emb(speakers)
            output = output + speaker_emb_s.unsqueeze(1).expand(
                -1, max_src_len, -1
            )
            speaker_emb_p = self.speaker_enc(speaker_meta)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        if self.use_jdit:
            if self.speaker_emb is not None:
                return (
                    output,
                    postnet_output,
                    p_predictions,
                    e_predictions,
                    log_d_predictions,
                    d_rounded,
                    src_masks,
                    mel_masks,
                    src_lens,
                    mel_lens,
                    mel_jdit,
                    alignments,
                    speaker_emb_p,
                    speaker_emb_s
                )
            else: 
                return (
                    output,
                    postnet_output,
                    p_predictions,
                    e_predictions,
                    log_d_predictions,
                    d_rounded,
                    src_masks,
                    mel_masks,
                    src_lens,
                    mel_lens,
                    mel_jdit,
                    alignments
                )
        else:
            if self.speaker_emb is not None:
                return (
                    output,
                    postnet_output,
                    p_predictions,
                    e_predictions,
                    log_d_predictions,
                    d_rounded,
                    src_masks,
                    mel_masks,
                    src_lens,
                    mel_lens,
                    speaker_emb_p,
                    speaker_emb_s
                )
            else:
                return (
                    output,
                    postnet_output,
                    p_predictions,
                    e_predictions,
                    log_d_predictions,
                    d_rounded,
                    src_masks,
                    mel_masks,
                    src_lens,
                    mel_lens
                )

    def speaker_gen(self, speaker_meta):
        with torch.no_grad():
            output = self.speaker_enc(speaker_meta)
            output = output.sample()
        return output

    def speaker_distribution(self, speaker_meta):
        with torch.no_grad():
            output = self.speaker_enc(speaker_meta)
        return output

    def synthesize_from_speaker_emb(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        accents=None,
        speaker_emb=None
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        output = self.encoder(texts, src_masks,accents=accents)
        if self.use_jdit:
            mel_jdit, gate_outputs, alignments = self.jdit(output, mels, src_lens)

        if self.speaker_emb is not None:
            output = output + speaker_emb.unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        if self.use_jdit:
            if self.speaker_emb is not None:
                return (
                    output,
                    postnet_output,
                    p_predictions,
                    e_predictions,
                    log_d_predictions,
                    d_rounded,
                    src_masks,
                    mel_masks,
                    src_lens,
                    mel_lens,
                    mel_jdit,
                    alignments,
                )
            else: 
                return (
                    output,
                    postnet_output,
                    p_predictions,
                    e_predictions,
                    log_d_predictions,
                    d_rounded,
                    src_masks,
                    mel_masks,
                    src_lens,
                    mel_lens,
                    mel_jdit,
                    alignments
                )
        else:
            if self.speaker_emb is not None:
                return (
                    output,
                    postnet_output,
                    p_predictions,
                    e_predictions,
                    log_d_predictions,
                    d_rounded,
                    src_masks,
                    mel_masks,
                    src_lens,
                    mel_lens,
                )
            else:
                return (
                    output,
                    postnet_output,
                    p_predictions,
                    e_predictions,
                    log_d_predictions,
                    d_rounded,
                    src_masks,
                    mel_masks,
                    src_lens,
                    mel_lens
                )


class SpeakerMetaEncoder(nn.Module):
    """
        TacoSpawn: https://arxiv.org/abs/2111.05095
        input: Speaker Metadata(a one-hot vector g+l-dim)
        output: GMM (K-mixtures D-dim. torch.Distribution)
        structure: dense neural net(-layer)
    """

    def __init__(self, preprocess_config, model_config) -> None:
        super(SpeakerMetaEncoder, self).__init__()
        self.model_config = model_config
        self.metadata_list = preprocess_config["speaker_generation"]["metadata"]
        self.input_dim = sum(list(map(lambda x: len(x), self.metadata_list.values())))
        self.K = self.model_config["speaker_generation"]["GMM_mixtures"]
        self.D = self.model_config["transformer"]["encoder_hidden"]

        self.pi_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.K),
            nn.Softmax(dim=1)
        )
        self.sigma_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.K*self.D),
            nn.Softplus()
        )
        self.mu_linear = nn.Linear(self.input_dim, self.K*self.D)

    def forward(self, input):
        pi = self.pi_linear(input).view(-1, self.K)
        sigma = self.sigma_linear(input).view(-1, self.K, self.D)
        mu = self.mu_linear(input).view(-1, self.K, self.D)

        mix = Categorical(pi)
        comp = Independent(Normal(loc=mu, scale=sigma), 1)
        gmm = MixtureSameFamily(mix, comp)

        return gmm