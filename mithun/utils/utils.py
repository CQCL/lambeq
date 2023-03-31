import os
import yaml
import pandas as pd
from mithun.utils import *

def read_config():
    filename = os.path.join(os.getcwd(), "mithun/utils/config.yml")
    with open(filename) as f :
        return yaml.safe_load(f)

def read_data_pandas(path,type_of_data="gold"):
    """Reads data from a csv file as a pandas dataframe:Schema, Event, Label
                    Args:
                        path: path to csv file
                    Returns:
                        List of schemas
                    """
    all_schemas=[]
    df=pd.read_csv(path,sep="\t")
    ls = OrderedDict(zip(df.index.tolist(), [''] * len(df.index.tolist()))).keys()
    ls = sorted(ls,reverse=True)
    for each_schema in ls:
        schema = MithunSchema(each_schema)
        full=df.loc[each_schema]
        schema.all = full["Event"].tolist()
        if type_of_data == "gold":
            schema.high = full.loc[lambda x: x["Label"] == 1]["Event"].tolist()
            schema.low = full.loc[lambda x: x["Label"] == 0]["Event"].tolist()
        else:
            if type_of_data == "train":
                all_events = []
                for row in full.iterrows():
                    event=MithunEvent(row[1]["Event"], row[1]["Likelihood"], row[1]["Normalized_Likelihood"])
                    all_events.append(event)
                schema.events=all_events
        all_schemas.append(schema)
    return all_schemas



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