import jsonlines
from mithun.utils.utils import *
claim_ev_label = {}



config = read_config()

with jsonlines.open(get_full_path(config['BASE_PATH_DATA'],config['SNLI_JSON_TRAIN_ORIGINAL'])) as f:
    for line in f.iter():
        key=(line["sentence1"], line["sentence2"])
        claim_ev_label[key] = line["gold_label"]

with open(get_full_path(config['BASE_PATH_DATA'],config['SNLI_TRAIN_LAMBEQ_FORMAT_CLAIM']), 'w') as claim_file , open(get_full_path(config['BASE_PATH_DATA'],config['SNLI_TRAIN_LAMBEQ_FORMAT_EVIDENCE']), 'w') as evidence_file:
    for cv,label in claim_ev_label.items():
        label_int=99
        match label:
            case 'neutral':
                label_int=0
            case 'contradiction' :
                label_int = 1
            case 'entailment' :
                label_int = 2
        claim_file.write(str(label_int)+"\t"+cv[0]+"\n")
        evidence_file.write(str(label_int) + "\t" + cv[1]+"\n")

    claim_file.close()
    evidence_file.close()