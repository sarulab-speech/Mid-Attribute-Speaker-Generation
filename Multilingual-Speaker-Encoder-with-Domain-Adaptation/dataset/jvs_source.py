# coding: utf-8

from __future__ import with_statement, print_function, absolute_import

from nnmnkwii.datasets import FileDataSource

import numpy as np
import os
from os import listdir
from os.path import join, splitext, isdir, exists, basename, isfile
from glob import glob
from collections import OrderedDict

available_speakers = ["jvs" + str(i).zfill(3) for i in range(1, 101)]
# Three lost wavfiles in this dataset
lost_wavfiles = {"jvs089": ["VOICEACTRESS100_019"],
                                "jvs030": ["VOICEACTRESS100_045"],
                                "jvs074": ["VOICEACTRESS100_094"]}


def _parse_speaker_info(data_root):
    # only use gender_f0range.txt
    speaker_info_path = join(data_root, "gender_f0range.txt")
    if not exists(speaker_info_path):
        raise RuntimeError("File {} doesn't exist".format(speaker_info_path))
    speaker_info = OrderedDict()
    terms = ["speaker", "Male_or_Female", "minf0[Hz]", "maxf0[Hz]"]
    with open(speaker_info_path, "r", encoding = "utf8") as file_:
        for line in file_:
            fields = line.strip().split()
            if fields[0] == terms[0]:
                continue
            assert len(fields) == 4
            speaker, gender, minf0, maxf0 = fields
            speaker_info[speaker] = {}
            speaker_info[speaker]["gender"] = gender
            speaker_info[speaker]["minf0"] = minf0
            speaker_info[speaker]["maxf0"] = maxf0
    return speaker_info

class _JVSBaseDataSource(FileDataSource):
    def __init__(self, data_root, speakers, labelmap, max_files):
        # only accept "jvs*" format
        self.data_root = data_root
        if speakers == "all":
            speakers = available_speakers
        for speaker in speakers:
            if speaker not in available_speakers:
                raise ValueError("Unknown speaker {}. It should be one of {}".format(speaker, available_speakers))
        self.speakers = speakers
        if labelmap is None:
            labelmap = {s: idx for idx, s in enumerate(speakers)}
        self.labelmap = labelmap
        self.labels = None
        self.max_files = max_files
        # TODO: currently there are many problem in falset, so we do not support falset
        self._folders = ["parallel100", "nonpara30",  "whisper10"]
        self._textfilename = "transcripts_utf8.txt"
        self._wavfoldername = "wav24kHz16bit"
        self.speaker_info = _parse_speaker_info(data_root)
        self._nonpara_no_wav = self._validate()

    def _validate(self, strict = False):
        # for each speaker, validate the data
        # TODO: check lab file
        nonpara_no_wav = {}
        for _, speaker in enumerate(self.speakers):
            # 1. check folder
            speaker_folder = join(self.data_root, speaker)
            folders = listdir(speaker_folder)
            nonpara_no_wav[speaker] = {}
            for folder in self._folders:
                assert folder in folders, "Can not find {} for {} in its directory {}".format(folder, speaker, speaker_folder)
            # 2. check wav and txt file,
            for folder in self._folders:
                speaker_textfile = join(speaker_folder, folder, self._textfilename)
                assert isfile(speaker_textfile), "File {} doesn't exist".format(speaker_textfile)
                speaker_wavfolder = join(speaker_folder, folder, self._wavfoldername)
                assert isdir(speaker_wavfolder), "Directory {} doesn't exist".format(speaker_wavfolder)
                # nonpara
                if folder == "nonpara30":
                    with open(speaker_textfile, encoding = "utf8") as file_:
                        txtfiles = [line.strip().split(":")[0] for line in file_]
                    wavfiles = listdir(speaker_wavfolder)
                    for file_ in txtfiles:
                        if file_ + ".wav" not in wavfiles:
                            nonpara_no_wav[speaker][file_] = '_'
                    assert (len(txtfiles) - len(nonpara_no_wav[speaker])) == len(wavfiles)
                # strict mode
                if strict:
                    with open(speaker_textfile, encoding = 'utf8') as file_:
                            txtlines = [line.strip().split(':')[0] for line in file_]
                    wavlines = [basename(line) for line in listdir(speaker_wavfolder)]
                    for line in txtlines:
                            if line+'.wav' not in wavlines:
                                    print(speaker, speaker_wavfolder, line)
        return nonpara_no_wav
        
    def collect_files(self, is_wav, nonpara = False, whisper = False): 
        paths, labels = [], []
        global lost_wavfiles
        max_files_per_speaker = self.max_files // len(self.speakers) if self.max_files else None
        for idx, speaker in enumerate(self.speakers):
            speaker_folder = join(self.data_root, speaker)

            def read_text(filepath, nonpara, para):
                with open(filepath, "r", encoding="utf8") as file_:
                    lines = [line.strip().split(":") for line in file_ if line.strip()]
                if nonpara:
                    lines = [line for line in lines if line[0] not in self._nonpara_no_wav[speaker]]
                if para and speaker in lost_wavfiles:
                    lines = [line for line in lines if line[0] not in lost_wavfiles[speaker]]
                lines.sort(key = lambda x: x[0])
                return [line[1] for line in lines]

            def read_wavs(folderpath):
                return sorted(glob(join(folderpath, "*.wav")), key = lambda x: basename(x))

            files = []
            for name, isset in zip(self._folders, [True, nonpara, whisper]):
                if isset:
                    folder = join(speaker_folder, name)
                    if is_wav:
                        files.extend(read_wavs(join(folder, self._wavfoldername)))
                    else:
                        files.extend(read_text(join(folder, self._textfilename), name == "nonpara30", name == "parallel100"))
            files = files[:max_files_per_speaker]
            paths.extend(files)
            labels.extend([self.labelmap[self.speakers[idx]]] * len(files))
        self.labels = np.array(labels, dtype=np.int16)
        return paths

class TranscriptionDataSource(_JVSBaseDataSource):
    """Transcription data source for JVS dataset

    The data source collects text transcriptions from JVS.
    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a transcription.

    Args:
        data_root (str): Data root.
        speakers (list): List of speakers to find. Speaker id must be ``str``.
          For supported names of speaker, please refer to ``available_speakers``
          defined in the module.
        labelmap (dict[optional]): Dict of speaker labels. If None,
          it's assigned as incrementally (i.e., 0, 1, 2) for specified
          speakers.
        max_files (int): Total number of files to be collected.

    Attributes:
        speaker_info (dict): Dict of speaker information dict. Keyes are speaker
          ids (str) and each value is speaker information consists of ``gender``,
          ``minf0`` and ``maxf0``.
        labels (numpy.ndarray): Speaker labels paired with collected files.
          Stored in ``collect_files``. This is useful to build multi-speaker
          models.

    """
    def __init__(self, data_root, speakers = available_speakers, labelmap = None, max_files = None):
        super(TranscriptionDataSource, self).__init__(data_root, speakers, labelmap, max_files)
    
    def collect_files(self, nonpara = False, whisper = False):
        return super(TranscriptionDataSource, self).collect_files(False, nonpara = nonpara, whisper=whisper)

class WavFileDataSource(_JVSBaseDataSource):
    """WavFile data source for JVS dataset.

    The data source collects text transcriptions from JVS.
    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a transcription.

    Args:
        data_root (str): Data root.
        speakers (list): List of speakers to find. Speaker id must be ``str``.
          For supported names of speaker, please refer to ``available_speakers``
          defined in the module.
        labelmap (dict[optional]): Dict of speaker labels. If None,
          it's assigned as incrementally (i.e., 0, 1, 2) for specified
          speakers.
        max_files (int): Total number of files to be collected.

    Attributes:
        speaker_info (dict): Dict of speaker information dict. Keyes are speaker
          ids (str) and each value is speaker information consists of ``gender``,
          ``minf0`` and ``maxf0``.
        labels (numpy.ndarray): Speaker labels paired with collected files.
          Stored in ``collect_files``. This is useful to build multi-speaker
          models.
    """
    def __init__(self, data_root, speakers = available_speakers, labelmap = None, max_files = None):
        super(WavFileDataSource, self).__init__(data_root, speakers, labelmap, max_files)
    
    def collect_files(self, nonpara = False, whisper = False):
        return super(WavFileDataSource, self).collect_files(True, nonpara = nonpara, whisper = whisper)

if __name__ == "__main__":
        source = "~/Downloads/jvs_ver1"
        from os.path import expanduser
        source = expanduser(source)
        td = TranscriptionDataSource(source, max_files = 5000) 
        wav = WavFileDataSource(source, max_files = 5000)
        import subprocess, random
        texts = td.collect_files(True, True)
        wavs = wav.collect_files(True, True)
        assert len(texts) == 5000
        assert len(wavs) == 5000
        comb = list(zip(texts, wavs))
        while True:
            t, w = random.choice(comb)
            print("Text: {}".format(t))
            print("display {} using sox".format(w))
            _ = subprocess.run("play {}".format(w), shell = True)
            x = input("Enter any key to next wav")
