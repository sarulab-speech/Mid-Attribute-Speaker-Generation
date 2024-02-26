import os
import json

path = "/path/to/jvs/gender_f0range.txt"

speakers = {}
with open(path, mode='r') as f:
    lines = f.readlines()
for i, line in enumerate(lines[1:]):
    spk, gen, _, _ = line.split(' ')
    speakers[spk] = [i, gen, "ja"]

with open("speakers.json", mode="w") as f:
    json.dump(speakers, f)