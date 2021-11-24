
import copy
import math
import hashlib
import random
from typing import Optional, List
from pycurator.common import paths
class Schema:
    """Store schemas and their events """
    def __init__(self, name : Optional[str],all:Optional[List[str]]=[],high:Optional[List[str]]=[],low:Optional[List[str]]=[]):
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

def read_data(path):
    """Reads data from a csv file. Data is expected to be in the format:
    schema_name, event, label
                    Args:
                        path: path to csv file
                    Returns:
                        List of schemas
                    """
    all_schema_names={}
    all_schemas=[]

    schema = Schema("dummy")
    with open(path) as f:
        for line in f:
            line=line.split("\t")
            schema_name=line[0]
            event=line[1]
            label=line[2]

            if schema_name not in all_schema_names:
                #very first time you encounter a new schema name,
                # add previous schema to list of schemas
                # todo: this is a temporary and nasty hack. load with pandas or something
                all_schemas.append(schema)

                #create new schema
                schema = Schema(schema_name)
                all_schema_names[schema_name]=1
                schema.all=[]
                schema.high = []
                schema.low = []


            schema.all.append(event)
            match int(label.strip()):
                case 1:
                    schema.high.append(event)
                case 0:
                    schema.low.append(event)
        all_schemas.append(schema)
    return all_schemas


class Baseline:

    """Base class for baseline methods."""
    def __init__(self,gold):
        self.gold=gold

    def calculate_metric(self,pred):
        """Compares the predicted data with gold data and calculates a metric (e.g.,Mean Average Precision)

        Args:
            gold: Gold data
            pred: Predicted data

        Returns:
            A score
        """

        return self.calculate_map(pred)
        #todo: add function for calculateNDCG. if not, just make it one cacluate_metric

    def calculate_map(self,pred):
        """Compares the predicted data with gold data and calculates a Mean Average Precision
        refer design document for details: https://docs.google.com/document/d/1JpCF5Yp5ybmosl6xyHCdcjewdkRjUBOLb2EyFpOKERA/edit?usp=sharing

               Args:
                   gold: Gold data
                   pred: Predicted data

               Returns:
                   A score
               """
        mean_avg_precision=0
        for schemag, schemap in (zip(self.gold,pred)):
            if not schemag.name=="dummy":
                precision_cumulative=0
                count_relevant_events=0
                for index,each_event in enumerate(schemap.high):
                    if each_event in schemag.high:
                        count_relevant_events+=1
                        precision_cumulative = precision_cumulative + (1/(index+1))
                avg_precision=precision_cumulative/count_relevant_events
                mean_avg_precision+=avg_precision

        return round(mean_avg_precision/(len(pred)-1),2) #todo- that minus 1 is for dummy. remove it once you move to pandas or something sensible to read csv

    def sort_data(self,gold):
        """Takes the gold data and resorts it with a given baseline method (e.g.,Sort by sentence length)

        Args:
            gold: Gold data

        Returns:
            Sorted gold data
        """

class CharLength(Baseline):

    def sort_data(self):
        """Takes the gold data.all_events and resorts it with length of event(number of characters)
        Then top n/2 becomes high_likelihood events and low n/2 becomes low_likelihood events where n is the total
        number of events in the schema

        Args:
            gold: Gold data

        Returns:
            Sorted gold data
        """

        to_sort=copy.deepcopy(self.gold)
        for schema in to_sort:
            if schema.name != "dummy":
                #todo temporary hack of reading files. this will go away once pandas comes in
                all_events=schema.all
                schema.all=sorted(all_events,key=lambda x: len(x),reverse=True)
                mid=math.ceil(len(schema.all)/2)
                schema.high= schema.all[:mid]
                schema.low = schema.all[mid:]
        return to_sort

    def calculate_metric(self):
        pred=self.sort_data()
        return Baseline.calculate_metric(self,pred)

    def __repr__(self):
        return "Length of events"

class Hash(Baseline):

    def sort_data(self):
        """Takes the gold data.all_events and resorts it with length of event(number of characters)
        Then top n/2 becomes high_likelihood events and low n/2 becomes low_likelihood events where n is the total
        number of events in the schema

        Args:
            gold: Gold data

        Returns:
            Sorted gold data
        """
        to_sort=copy.deepcopy(self.gold)
        for schema in to_sort:
            if schema.name != "dummy":
                #todo temporary hack of reading files. this will go away once pandas comes in
                all_events=schema.all
                schema.all=sorted(all_events,key=lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest(),reverse=True)
                mid=math.ceil(len(schema.all)/2)
                schema.high= schema.all[:mid]
                schema.low = schema.all[mid:]
        return to_sort

    def calculate_metric(self):
        pred=self.sort_data()
        return Baseline.calculate_metric(self,pred)

    def __repr__(self):
        return "hash of events"

class Random(Baseline):

    def sort_data(self):
        """Takes the gold data.all_events and resorts it with length of event(number of characters)
        Then top n/2 becomes high_likelihood events and low n/2 becomes low_likelihood events where n is the total
        number of events in the schema

        Args:
            gold: Gold data

        Returns:
            Sorted gold data
        """
        to_sort=copy.deepcopy(self.gold)
        for schema in to_sort:
            if schema.name != "dummy":
                #todo temporary hack of reading files. this will go away once pandas comes in
                all_events=schema.all
                random.seed(10)
                schema.all=sorted(all_events,key=lambda x: random.random(),reverse=True)
                mid=math.ceil(len(schema.all)/2)
                schema.high= schema.all[:mid]
                schema.low = schema.all[mid:]
        return to_sort

    def calculate_metric(self):
        pred=self.sort_data()
        return Baseline.calculate_metric(self,pred)

    def __repr__(self):
        return "Random number generation"

if __name__=="__main__":
    gold=read_data("data/gold.tsv")
    for obj in CharLength(gold),Hash(gold),Random(gold):
        print(f"the mean average precsion calculated using {obj} is {obj.calculate_metric()}")
