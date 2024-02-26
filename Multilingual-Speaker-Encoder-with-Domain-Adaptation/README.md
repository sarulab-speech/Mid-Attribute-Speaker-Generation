# Multilingual-Speaker-Encoder-with-Domain-Adaptation
This repo contains a multilingual speaker encoder for **English** and **Japanese**.
The model uses [Generalized End-to-End loss](https://arxiv.org/abs/1710.10467) for speaker verification and domain adaptation to solve the domain-shift problem.

## Setup
Follow the procedure below to setup you environment
1. Download JVS and VCTK or your own dataset
2. Set the dataset paths in `config/config.yaml`, set the config path in `hparam.py`
3. (optional) The original code only supports VCTK and JVS, if you use your own dataset, you need to write your own dataset class under `dataset` directory.
3. `python3 data_preprocess.py`

## Train
`python3 train_speech_embedder.py`
This will train the baseline model without domain adaptation.
For more settings please refer to `config/config.yaml`.

## Citation
Please kindly cite this repo and the following paper if you find this code useful.
```
@inproceedings{xin21_interspeech,
  author={Detai Xin and Yuki Saito and Shinnosuke Takamichi and Tomoki Koriyama and Hiroshi Saruwatari},
  title={{Cross-Lingual Speaker Adaptation Using Domain Adaptation and Speaker Consistency Loss for Text-To-Speech Synthesis}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={1614--1618},
  doi={10.21437/Interspeech.2021-897}
}
```
## Licence
MIT
