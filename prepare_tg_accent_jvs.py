import argparse
import pathlib
from pathlib import Path
import re
import sys
import yaml

from tqdm import tqdm

from convert_label_jvs import read_lab


# full context label to accent label from ttslearn
def numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))
def pp_symbols(labels, drop_unvoiced_vowels=True):
    PP = []
    accent = []
    N = len(labels)

    for n in range(len(labels)):
        lab_curr = labels[n]


        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        if p3 == 'sil':
            assert n== 0 or n == N-1
            if n == N-1:
                e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    PP.append("")
                elif e3 == 1:
                    PP.append("")
            continue
        elif p3 == "pau":
            PP.append("sp")
            accent.append('0')
            continue
        else:
            PP.append(p3)
        # アクセント型および位置情報（前方または後方）
        a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
        # アクセント句におけるモーラ数
        f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)
        lab_next = labels[n + 1]
        a2_next = numeric_feature_by_regex(r"\+(\d+)\+", lab_next)
        # アクセント境界
        if a3 == 1 and a2_next == 1:
            accent.append("#")
        # ピッチの立ち下がり（アクセント核）
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            accent.append("]")
        # ピッチの立ち上がり
        elif a2 == 1 and a2_next == 2:
            accent.append("[")
        else:
            accent.append('0')
    return PP, accent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")

    args = parser.parse_args()

    # create output directory
    
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    corpus_path = Path(config["path"]["corpus_path"])
    preprocessed_path = Path(config["path"]["preprocessed_path"])
    with_accent = config["preprocessing"]["accent"]["use_accent"]

    tg_dir = (preprocessed_path / 'TextGrid')
    ac_dir = (preprocessed_path/ 'accent')
    if not tg_dir.exists():
        tg_dir.mkdir(parents=True)
    if not ac_dir.exists():
        ac_dir.mkdir()


    speakers = ["jvs{:0>3d}".format(i) for i in range(1, 101)]
    for speaker in speakers:
        lab_paths = [(corpus_path / speaker / 'parallel100' / 'lab' / 'ful'), (corpus_path / speaker / 'nonpara30' / 'lab' / 'ful')]
        lab_folders = [lab_path.glob('**/*.lab') for lab_path in lab_paths]
        tg_dir_speaker = (tg_dir / speaker)
        if not tg_dir_speaker.exists():
            tg_dir_speaker.mkdir() 
        for lab_files in lab_folders:
            # iter through lab files
            for lab_file in tqdm(lab_files):
                if with_accent:
                    accent = []
                    with open(lab_file) as f:
                        lines = f.readlines()
                    lab, accent = pp_symbols(lines)
                    if not (ac_dir / lab_file.with_suffix('.accent')).exists():
                        with open(ac_dir/ lab_file.with_suffix('.accent').name,mode='w') as f:
                            f.writelines([''.join(accent)])        
                label = read_lab(str(lab_file))
                textgridFilePath = tg_dir_speaker/lab_file.with_suffix('.TextGrid').name
                label.to_textgrid(textgridFilePath)

        

    
