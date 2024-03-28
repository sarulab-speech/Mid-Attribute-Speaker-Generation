# FastSpeech2 implementation with Mid-attribute speaker generation

Forked from https://github.com/Wataru-Nakata/FastSpeech2-JSUT

GE2E module from https://github.com/Aria-K-Alethia/Multilingual-Speaker-Encoder-with-Domain-Adaptation/tree/main

TacoSpawn: https://arxiv.org/abs/2111.05095

Mid-attribute speaker generation: https://arxiv.org/abs/2210.09916

## How to setup
### Download corpora
JSUT: https://sites.google.com/site/shinnosuketakamichi/publication/jsut
JVS: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
VCTK: http://www.udialogue.org/ja/download-ja/cstr-vctk-corpus.html

### Setup environment
```
git submodule update --init
unzip hifigan/generator_universal.pth.tar.zip -d hifigan/

# Setup virtual environment before pip install if you want
pip install -r requirements.txt
```

### Do preprocessing
Before preprocessing, overwrite corpus_path of preprocess_*.yaml.
> In this version, a configuration for using the GE2E loss is necessary. Therefore, instead of using the config/JVS-VCTK configuration, you should use any of the configurations under config/JVS-VCTK_langemb_configs.

#### Prepare TextGrid
For JSUT
```
mkdir -p raw_data/JSUT/JSUT
cp path/to/JSUT/*/wav/*.wav raw_data/JSUT/JSUT
python retriever/retrieve_transcripts_jsut.py
python prepare_tg_accent_jsut.py jsut-lab/ preprocessed_data/JSUT/ JSUT --with_accent True
```

For JVS
> The prepare_tg_accent_jvs.py script modifies the time formats of .lab.
```
mkdir -p raw_data/JVS
python retriever/retrieve_jvs.py
python prepare_tg_accent_jvs.py config/JVS-VCTK/
```

For VCTK
> You have to prepare .lab by yourself.
> If you want to use prepare_tg_hts.py, you should prepare HTK/HTS-style .lab with the directory structure below:
```
.
└ lab
  └ VCTK
      ├ p225(speaker ID)
      |  └ labels
      |     ├ p225_001(utterance ID).lab
      |     ├ p225_002.lab
      |     ⋮
      |     └ p225_366.lab
      ├ p226/
      ⋮
      └ p376/
```

```
mkdir -p raw_data/VCTK
python retriever/retrieve_vctk.py
python prepare_tg_accent_hts.py config/JVS-VCTK VCTK
```

#### Prepare other features (pitch, duration, energy)
```
python preprocess.py config/JVS-VCTK
```

### Train
```
python train.py config/JVS-VCTK
```

### Synthesize from existent speaker
```
python3 synthesize.py --text "音声合成、たのちい" --speaker_id 0 --restore_step 20000 --mode single -c config/JVS-VCTK
```

### Synthesize from non-existent speaker
Under construction (examples_gen.py may help you)


## Memo

JVS has some wrong alignments. Remove or fix them before training.

- jvs070-VOICEACTRESS100_001
