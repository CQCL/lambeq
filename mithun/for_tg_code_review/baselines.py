
import copy
import hashlib
import random
import argparse
from pycurator.common.logger import return_logger
from typing import Optional, List
from pycurator.gpt2_component.filter import BertRanker
from collections import OrderedDict
import pandas as pd
from sdf.yaml_schema import Schema
from pathlib import Path
from tqdm import tqdm
logging = return_logger("gpt2_component")
from pycurator.gpt2_component.bert_ranking import (
    get_cross_claim_evidence_percentage_attention_weights,
)

RUN_EVALUATION=True
GENERATE_TRAINING_DATA=True
# todo: store in config
ALEX_DATA_FILE = "pycurator/data/likelihood/alex_freq_based_approach.tsv"
GOLD_DATA_FILE = "pycurator/data/likelihood/gold.tsv"
OUT_FILE_TRAINING_DATA = "pycurator/data/likelihood/out.tsv"
SCHEMA_DIR = "pycurator/data/likelihood/schema_dir"
TEST_SCHEMA_DIR = "pycurator/data/likelihood/test_schema_dir"
DEFAULT_BERT_MODEL = "bert-base-uncased"

class MithunEvent:
    """Store schemas and their events """


    def __init__(self, name ="", likelihood=0, normalized_likelihood=0):
        """Initialize class

                Args:

                Returns:
                    An instance of class Schema
                """
        self.name=name
        self.likelihood=likelihood
        self.normalized_likelihood= normalized_likelihood


class MithunSchema:
    """Store schemas and their events """
    def __init__(self, name : Optional[str], all:Optional[List[str]]=[], high:Optional[List[str]]=[], low:Optional[List[str]]=[], events:Optional[List[MithunEvent]]=[], premise:Optional[str]=""):
        """Initialize class

                Args:
                    all: all events
                    high: events with high likelihood, label=1
                    low: events with low likelihood, label=0

                Returns:
                    An instance of class Schema
                """
        self.name=name
        self.all=all
        self.high=high
        self.low=low
        self.events=events
        self.premise=premise

def write_schema_disk(self,schemas: List[MithunSchema]):
    pass

def read_gold_data():
    """Function created to be used only in test cases
                       Args:

                       Returns:
                           data read from gold
                       """
    return read_data_pandas(GOLD_DATA_FILE)


def read_data_pandas(path,type_of_data="gold"):
    """Reads data from a csv file as a pandas dataframe:Schema, Event, Label
                    Args:
                        path: path to csv file
                    Returns:
                        List of schemas
                    """
    all_schemas=[]
    df=pd.read_csv(path,sep="\t", index_col="Schema")
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



class Baseline:

    """Base class for baseline methods."""
    def __init__(self,gold):
        logging.debug("inside Baseline.init")
        self.gold=gold
        self.to_sort = copy.deepcopy(self.gold)

    def sort_data(self, key_line, to_sort, sorted_list=None):
        """if a sorted_list is not given Takes the gold data and resorts it with a given baseline method's formula for sorting
        (e.g.,Sort by sentence length). i.e sort based on key_line after which top n/2 becomes
        high_likelihood events and low n/2 becomes low_likelihood events where n is the total
        number of events in the schema

        Args:
            key_line: the formula that is used to sort

        Returns:
            Sorted gold data
        """
        logging.debug("inside Baseline.sort_data")
        logging.debug(f"value of sorted_list is {sorted_list}")
        if sorted_list is None:
            logging.debug(f"value of key_line is {key_line}")

            for schema in to_sort:
                original_schema=copy.deepcopy(schema)
                all_events = schema.all
                schema.all = sorted(all_events, key=key_line, reverse=True)
                # from the sorted list pick top n as high likelihood where n is the number of original high likelihood
                # events in  the gold data
                schema.high = schema.all[:len(original_schema.high)]
                schema.low = schema.all[len(original_schema.high):]
            return to_sort
        else:
            return sorted_list

    def calculate_metric(self, gold, data_to_calculate:Optional[List[MithunSchema]]):
        """Compares the predicted data with gold data and calculates a metric (e.g.,Mean Average Precision)
            Keeping this as a wrapper class just in case tomorrow the metric changes (e.g. NDCG)
        Args:
            gold: Gold data
            data_to_calculate: Predicted data

        Returns:
            A score
        """
        logging.debug("inside Baseline.calculate_metric")
        return self.calculate_map(gold, data_to_calculate)


    def calculate_map(self,gold,pred):
        """Compares the predicted data with gold data and calculates a Mean Average Precision
        refer design document for details: https://docs.google.com/document/d/1JpCF5Yp5ybmosl6xyHCdcjewdkRjUBOLb2EyFpOKERA/edit?usp=sharing

               Args:
                   gold: Gold data
                   pred: Predicted data

               Returns:
                   A score
               """
        assert len(gold) == len(pred)
        cumulative_avg_precision=0

        for schemag, schemap in (zip(gold,pred)):
            avg_precision = 0
            assert schemag.name.lower() == schemap.name.lower()
            assert len(schemag.high) == len(schemap.high)
            precision_cumulative=0
            count_relevant_events=0
            for index,each_event in enumerate(schemap.high):
                if each_event in schemag.high:
                    count_relevant_events+=1
                    precision_cumulative = precision_cumulative + (1/(index+1))
            if count_relevant_events > 0:
                avg_precision=precision_cumulative/count_relevant_events
            cumulative_avg_precision+=avg_precision
        return round(cumulative_avg_precision/len(pred),2)



class ByCharLength(Baseline):

    def sort_data(self):
        """Prepare keyline used to be used in sorting and calls baseclass.sort_data with it which in this case is
        length of sentence/number of characters in an ebent
               Args:

               Returns:
                   Sorted gold data
               """
        logging.debug("inside CharLength.sort_data")
        keyline=lambda x: len(x)
        return Baseline.sort_data(self,keyline,self.to_sort)

    def calculate_metric(self) -> float:
        """Take to_sort data, rank it, calculate metric, return metric value (e.g., Mean Average Precision)
               Args:
                   to_sort: the dataset to sort. If none is passed, the gold data will be used as to_sort- e.g., in
                   case of random baselines

               Returns:
                   Sorted  data
               """

        logging.debug("inside calculate_metric CharLength")
        pred=self.sort_data()
        logging.debug(f"value of pred is {pred}")
        return Baseline.calculate_metric(self,self.gold,pred)

    def __repr__(self):
        return "Length of events"

class ByHash(Baseline):

    def sort_data(self):
        """Prepare keyline used to be used in sorting and calls baseclass.sort_data with it which in this case is
        hash of the sentence of the event
               Args:

               Returns:
                   Sorted gold data
       """
        keyline = lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest()
        return Baseline.sort_data(self,keyline,self.to_sort)

    def calculate_metric(self):
        pred=self.sort_data()
        return Baseline.calculate_metric(self,self.gold,pred)

    def __repr__(self):
        return "hash of events"

class ByRandomNumber(Baseline):

    def sort_data(self):
        """Prepare keyline used to be used in sorting and calls baseclass.sort_data with it which in this case is
        a random number generated with known seed (to create repeatable results)
               Args:

               Returns:
                   Sorted gold data
       """
        random.seed(10)
        keyline = lambda x: random.random()
        return Baseline.sort_data(self,keyline,self.to_sort)


    def calculate_metric(self):
        pred=self.sort_data()
        return Baseline.calculate_metric(self,self.gold,pred)

    def __repr__(self):
        return "Random number generation"



class BertAsRanker(Baseline):

    def __init__(self,gold):
        logging.debug("inside BertAsRanker.init")
        self.gold=gold


    def sort_data(self, premises):
        """Prepare keyline used to be used in sorting and calls baseclass.sort_data with it which in this case is
        length of sentence/number of characters in an ebent
               Args:

               Returns:
                   Sorted gold data
               """
        logging.debug("inside BertAsRanker.sort_data")
        list_schemas=[]
        bert_ranker = BertRanker()
        for schema in tqdm(premises,desc="Attach attention scores:",total=len(premises)):
            original=copy.deepcopy(schema)
            logging.debug(f"value of f.name is {schema.name}")
            logging.debug(f"value of f.all is {schema.all}")
            schema.all, scores = bert_ranker.meet_criteria_using_same_sequence(schema.all)
            # from the sorted list pick top n as high likelihood where n is the number of original high likelihood
            # events in  the gold data
            all_events=[]
            for each_event_text in schema.all:
                each_event= MithunEvent(each_event_text,scores[each_event_text])
                all_events.append(each_event)
            schema.events=all_events
            schema.high = schema.all[:len(original.high)]
            schema.low = schema.all[len(original.high):]
            list_schemas.append(schema)
        assert len(list_schemas) > 0
        return list_schemas

    def __repr__(self):
        return "BertAsRanker"



    def calculate_metric(self):
        logging.debug("inside BertAsRanker.calculate_metric")
        #in its base form this function compares each schema concatenated to each event in that schema- and returns attention score
        pred=self.sort_data(self.gold)
        return Baseline.calculate_map(self,self.gold,pred)

class UsingPreDefinedLikelihoodValue(Baseline):
    """Given schemas and their events and likelihoods calculate metric """
    def sort_data(self, to_sort):
        """For train data sort by likelihood value

        Args:
            key_line: the formula that is used to sort

        Returns:
            Sorted gold data
        """
        for schemag,schemat in zip(self.gold,to_sort):
            assert schemag.name.lower()==schemat.name.lower()
            original_schema = schemag
            all_events = [x for x in schemat.events]
            s = sorted(all_events, key=lambda x: x.likelihood, reverse=True)
            schemat.all = [x.name for x in s]
            # from the sorted list pick top n as high likelihood where n is the number of original high likelihood
            # events in  the gold data
            schemat.high = schemat.all[:len(original_schema.high)]
            schemat.low = schemat.all[len(original_schema.high):]
        return to_sort


    def calculate_metric(self):
        data_to_calculate = read_data_pandas(ALEX_DATA_FILE, type_of_data="train")
        pred=self.sort_data(data_to_calculate)
        logging.debug(f"value of pred is {pred}")
        return Baseline.calculate_metric(self,self.gold,pred)

    def __repr__(self):
        return "Likelihood value"

class TrainingData:
    def __init__(self):
        pass
    def create_positive_examples(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--schema-dir", default=TEST_SCHEMA_DIR, type=Path, help="Schema directory.")
        args = parser.parse_args()
        schema_dir: Path = args.schema_dir

        if not schema_dir.is_dir():
            raise NotADirectoryError(
                f"Schema directory {schema_dir} does not exist or is not a directory."
            )
        schema_paths = sorted(schema_dir.glob("*.yaml"))
        schemas = [
            Schema.load(schema_path) for schema_path in tqdm(schema_paths, desc="Loading schemas")
        ]
        all_schemas = []
        for schema_from_file in schemas:
            new_schema = MithunSchema(name=schema_from_file.description)
            all_events = []
            for step in schema_from_file.steps:
                event = MithunEvent(name=step.id)
                all_events.append(event)
            if len(all_events) > 0:
                new_schema.all = [x.name for x in all_events]
                new_schema.premise = ". ".join(new_schema.all)
                new_schema.events = all_events
            all_schemas.append(new_schema)
        brt=BertAsRanker(all_schemas)
        all_schemas=brt.sort_data(all_schemas)
        return all_schemas

    def write_to_disk(self,schemas):
        with open(OUT_FILE_TRAINING_DATA,'w') as f:
            f.write("Event\tLabel\tLikelihood\n")
        with open(OUT_FILE_TRAINING_DATA,'a') as f:
            for schema in schemas:
                for event in schema.events:
                    f.write(f"{schema.premise}\t{event.name}\t{event.likelihood}\n")

    def generate_training_data(self):
        pos=self.create_positive_examples()
        self.write_to_disk(pos)

def main():
    if(RUN_EVALUATION):
        gold = read_data_pandas(GOLD_DATA_FILE)
        list_type_of_baselines=[ByCharLength(gold), ByHash(gold), ByRandomNumber(gold), UsingPreDefinedLikelihoodValue(gold), BertAsRanker(gold)]
        # list_type_of_baselines = [BertAsRanker(gold)]
        for obj in list_type_of_baselines:
            logging.info(f"**************the mean average precision calculated using {obj} is {obj.calculate_metric()}")

    if(GENERATE_TRAINING_DATA):
        tr = TrainingData()
        tr.generate_training_data()



if __name__=="__main__":
    main()
