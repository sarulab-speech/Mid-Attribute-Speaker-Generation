import argparse
from pathlib import Path
import yaml

from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config")
    parser.add_argument("--corpus", type=str, default=None, help="corpus name")
    args = parser.parse_args()

    preprocess_config = yaml.load(open(args.config + "/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    preprocess_config["pitch"]["normalization"] = False
    preprocess_config["energy"]["normalization"] = False
    if args.corpus != None:
        corpus_config = yaml.load(open(args.config + "/preprocess_" + args.corpus + ".yaml", "r"), Loader=yaml.FullLoader)
        config = corpus_config
        config["preprocessing"] = preprocess_config
        config["preprocessing"]["text"] = corpus_config["text"]
        configs = [config]
    else:
        configs = []
        for corpus in Path(args.config).glob("preprocess_*.yaml"):
            corpus_config = yaml.load(open(corpus, "r"), Loader=yaml.FullLoader)
            config = corpus_config
            config["preprocessing"] = preprocess_config
            config["preprocessing"]["text"] = corpus_config["text"]
            configs.append(config)

    for config in configs:
        print("preprocessing...:", config["dataset"])
        preprocessor = Preprocessor(config)
        preprocessor.build_from_path()
