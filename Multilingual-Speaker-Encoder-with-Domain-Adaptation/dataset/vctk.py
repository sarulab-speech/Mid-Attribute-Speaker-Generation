import os
from os.path import basename, join, exists, expanduser
from glob import glob
import numpy as np
from hparam import hparam as hp
from collections import OrderedDict
from nnmnkwii.datasets import vctk

class VCTK:
    _name = 'vctk'
    _language = 'en'

    def __init__(self, root, test_speakers):
        self.root = expanduser(root)
        self.wav_source = vctk.WavFileDataSource(self.root)
        self.spkrs = self.collect_spkr() 
        self.spkr2wavs = self.get_spkr2wavs()
        for s in list(self.spkrs.keys()):
            if s not in self.spkr2wavs:
                self.spkrs.pop(s)
        self.test_speakers = test_speakers

    def get_spkr2wavs(self):
        wavs = self.wav_source.collect_files()
        spkr2wavs = OrderedDict()
        label2spkr = {v: k for k, v in self.wav_source.labelmap.items()}
        for label, wav in zip(self.wav_source.labels, wavs):
             spkr = 'p'+label2spkr[label]
             if spkr not in spkr2wavs:
                spkr2wavs[spkr] = []
             spkr2wavs[spkr].append(wav)
        return spkr2wavs

    def collect_spkr(self):
        spkrs = self.wav_source.speaker_info.copy()
        spkrs = {'p'+k: v for k, v in spkrs.items()}
        return spkrs

    @property
    def spkr_number(self):
        return len(self.spkrs)

    def get_gender(self, spkr):
       return self.spkrs[spkr]['GENDER'].lower()

    def spkr_wavlist(self):
        for spkr, wavlist in self.spkr2wavs.items():
            yield spkr, wavlist

    @staticmethod
    def collect_feat(speaker_list):
        raise NotImplementedError()
        paths = ['GE2E/train_tisv', 'GE2E/test_tisv']
        spkr2feat = {}
        for s in speaker_list:
            feats = []
            for path in paths:
                feats.extend(glob(join(path, 'vctk_{}*'.format(s))))
            if not feats:
                print('speaker {} has no feat'.format(s))
                continue
            buf = [np.load(feat) for feat in feats]
            feat = np.concatenate(buf, axis=0)
            feat = np.transpose(feat, axes=(0,2,1))
            spkr2feat[s] = feat
        return spkr2feat
    
if __name__ == '__main__':
    from os.path import expanduser
    root = expanduser('~/Downloads/VCTK-Corpus/')
    jvs = JVS(root)
    print(jvs.spkr_number)
    print(jvs.spkrs)
    out = [(s, wavlist) for s, wavlist in jvs.spkr_wavlist()]
    print(out[0], len(out[0][1]))
