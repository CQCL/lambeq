import jsonlines
from configs import *
claim_ev_label = {}

with jsonlines.open(MNLI_JSON_TRAIN) as f:
    for line in f.iter():
        key=(line["sentence1"], line["sentence2"])
        claim_ev_label[key] = line["gold_label"]

print(claim_ev_label)

