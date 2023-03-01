import jsonlines
from configs import *
with jsonlines.open(MNLI_JSON_TRAIN) as f:
    for line in f.iter():
        print(line)