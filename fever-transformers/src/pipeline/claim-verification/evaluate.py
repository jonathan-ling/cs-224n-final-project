#!/usr/bin/env python3

import argparse
import json
import os

from collections import defaultdict

from fever.scorer import fever_score
from prettytable import PrettyTable

def main(prediction_file, golden_file):
    path = os.getcwd()
    prediction_file = os.path.join(path, prediction_file)
    golden_file = os.path.join(path, golden_file)

    actual = []
    with open(golden_file, "r") as f:
        for line in f:
            actual.append(json.loads(line))

    predictions = []
    with open(prediction_file, "r") as f:
        for i,line in enumerate(f):
            prediction = json.loads(line)
            evidence = list(filter(lambda e: e[0] != None, map(lambda e: e[2:], actual[i]["evidence"][0])))
            prediction["predicted_evidence"] = evidence
            predictions.append(prediction)

    assert len(predictions) == len(actual), "The two file provided does not have the same number of lines"

    score,acc,_,_,_ = fever_score(predictions, actual)

    tab = PrettyTable()
    tab.field_names = ["OFEVER Score", "Label Accuracy"]
    tab.add_row((round(score,4),round(acc,4)))
    print(tab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", type=str, help="input dataset")
    parser.add_argument("--golden-file", type=str, help="original input dataset with gold predictions")
    args = parser.parse_args()
    main(args.prediction_file, args.golden_file)
