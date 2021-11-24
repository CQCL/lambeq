import os
import yaml
from mithun.utils import *

def read_config():
    filename = os.path.join(os.getcwd(), "mithun/utils/config.yml")
    with open(filename) as f :
        return yaml.safe_load(f)



def read_data_float_label(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            line_split=line.split("\t")
            t = float(line_split[0])
            labels.append([t])
            sentences.append(line_split[1].strip().lower())
    return labels, sentences


def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = float(line[0])
            labels.append([t, 1-t])
            sentences.append(line[1:].strip())
    return labels, sentences



def read_data_string_label(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            line_split=line.split("\t")
            labels.append([line_split[0]])
            sentences.append(line_split[1].strip())
    return labels, sentences


def get_full_path(dir, filename):
    return os.path.join(os.getcwd(),dir, filename)