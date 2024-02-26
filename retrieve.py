import subprocess
from pathlib import Path
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to config folder")
parser.add_argument("--corpus", type=str, required=True, help="name of corpus")
args = parser.parse_args()

corpus = args.corpus.lower()
retriever_path = Path("retriever") / ("retrieve_" + corpus + ".py")
if retriever_path.exists():
    cp = subprocess.run(["python", retriever_path, args.config])
else:
    print("Error: retriever doesn't exist for " + corpus, file=sys.stderr)
    sys.exit(1)



