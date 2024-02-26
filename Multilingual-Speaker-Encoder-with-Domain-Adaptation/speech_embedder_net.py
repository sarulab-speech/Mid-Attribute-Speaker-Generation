#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
import random

from .hparam import hparam as hp
from .utils import get_similarity, get_contrast_loss, get_softmax_loss, count_label

from .module import grad_reverse, MultiLayerNN, LinearNorm, ConvNorm2D, ConvNorm1D, ResBlock

class LSTMEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LSTMEncoder, self).__init__()  
        pass

class ResCNN(nn.Module):
    def __init__(self):
        super(ResCNN, self).__init__()
        in_channels = 1
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        # rescnn 
        sc = 16
        self.conv1 = ConvNorm2D(in_channels, sc, 5, 2, w_init_gain='relu')
        self.res1 = ResBlock(sc, sc)
        self.conv2 = ConvNorm2D(sc, sc * 2, 5, 2, w_init_gain='relu')
        sc *= 2
        self.res2 = ResBlock(sc, sc)
        self.conv3 = ConvNorm2D(sc, sc * 2, 5, 2, w_init_gain='relu')
        sc *= 2
        self.res3 = ResBlock(sc, sc)
        self.conv4 = ConvNorm2D(sc, sc * 2, 5, 2, w_init_gain='relu')
        sc *= 2
        self.res4 = ResBlock(sc, sc)
        self.avgpool = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_layer = nn.Sequential(
            self.conv1,
            self.relu,
            self.res1,
            self.conv2,
            self.relu,
            self.res2,
            self.conv3,
            self.relu,
            self.res3,
            self.conv4,
            self.relu,
            self.res4
        )
    def forward(self, x):
        #self.lstm.flatten_parameters()
        #x, _ = self.lstm(x) 
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)
        x = self.conv_layer(x)
        batch_size = x.shape[0]
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(batch_size, -1) # 5*sc
        return x

class SpeechEmbedder(nn.Module):
    
    def __init__(self):
        super(SpeechEmbedder, self).__init__()    
        dropout = 0
        if hp.model.architecture == 'LSTM':
            self.LSTM_stack = nn.LSTM(hp.data.nmels, hp.model.hidden,
                            num_layers=hp.model.num_layer,
                            batch_first=True, dropout=dropout)
            for name, param in self.feat_extractor.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
        elif hp.model.architecture == 'rescnn':
            self._feat_extractor = ResCNN()
        self.projection = LinearNorm(hp.model.hidden, hp.model.proj)
        if hp.model.da:
            outdim = count_label(hp)
            self.da_classifier = Classifier(hp.model.proj, outdim)
        else:
            self.da_classifier = None
        self.dropout = nn.Dropout(dropout)


    @property
    def feat_extractor(self):
        if hp.model.architecture == 'LSTM':
            return self.LSTM_stack
        else:
            return self._feat_extractor
        
    def da_parameters(self):
        return list(self.da_classifier.parameters())

    def main_parameters(self):
        return list(self.feat_extractor.parameters()) + list(self.projection.parameters())

    def adjust_da_coef(self, p):
        self.da_classifier.adjust_da_coef(p)

    def compute_embedding(self, x):
        if hp.model.architecture == 'LSTM':
            self.feat_extractor.flatten_parameters()
            x, _ = self.feat_extractor(x.float()) #(batch, frames, n_mels)
            #only use last frame
            x = x[:, -1]
        else:
            x = self.feat_extractor(x)
        x = self.dropout(x)
        spkr_emb = self.projection(x)
        spkr_emb = spkr_emb / torch.norm(spkr_emb, dim=1, keepdim=True)
        out_dict = {'embeddings': spkr_emb}
        return out_dict
        
    def forward(self, x, detach=False):
        # detach should be set only when we need to train classifier of da
        if hp.model.architecture == 'LSTM':
            self.feat_extractor.flatten_parameters()
            x, _ = self.feat_extractor(x.float()) #(batch, frames, n_mels)
            #only use last frame
            x = x[:, -1]
        else:
            x = self.feat_extractor(x)
        x = self.dropout(x)
        spkr_emb = self.projection(x)
        spkr_emb = spkr_emb / torch.norm(spkr_emb, dim=1, keepdim=True)
        out_dict = {}
        if hp.model.da:
            if detach:
                spkr_emb = spkr_emb.detach()
            da_lang_logits = self.da_classifier(spkr_emb)
            out_dict['da_lang_logits'] = da_lang_logits
        
        out_dict['embeddings'] = spkr_emb
        return out_dict

class Classifier(nn.Module):
    def __init__(self, indim, outdim):
        super(Classifier, self).__init__()
        self.classifier = MultiLayerNN(indim, [indim, indim, outdim], dropout=0.2)
        #self.classifier = LinearNorm(indim, outdim, w_init_gain='sigmoid')
        self.coef = 0

    def adjust_da_coef(self, p):
        if p <= hp.model.da_startpoint:
            self.coef = 0
        else:
            self.coef = 2 / (1 + math.exp(-10*p)) - 1
            #self.coef = 1 

    def forward(self, embeddings):
        # embeddings = grad_reverse(embeddings, self.coef)
        logits = self.classifier(embeddings).squeeze(1)
        return logits

class GE2ELoss(nn.Module):
    
    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
        self.device = device
        if count_label(hp) == 1:
            self.da_loss_funct = nn.BCEWithLogitsLoss(reduction='sum')
        else:
            self.da_loss_funct = nn.CrossEntropyLoss(reduction='sum')
        
    def forward(self, embeddings, lang_logits, langs, **kwargs):
        torch.clamp(self.w, 1e-6)
        similarity = get_similarity(embeddings)
        #print(similarity[0, 0, :])
        similarity = self.w * similarity + self.b
        if hp.model.loss == 'contrast':
            loss = get_contrast_loss(similarity)
        else:
            loss = get_softmax_loss(similarity)

        if lang_logits is not None:
            da_loss = self.da_loss_funct(lang_logits, langs)
        else:
            da_loss = 0
        return loss + da_loss, loss, da_loss
