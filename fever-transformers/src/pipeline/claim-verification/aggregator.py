#!/usr/bin/env python3

import argparse
import bisect
import csv
import json
import os

from collections import defaultdict
from functools import reduce

from tqdm import tqdm

# Code below is adapted from run.py, as well as
# https://github.com/takuma-ynd/fever-uclmr-system/blob/interactive/neural_aggregator.py

import sys
from collections import Counter
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

import pandas as pd
import pickle

class Net(nn.Module):
    def __init__(self, hidden_size = 100):
        super(Net, self).__init__()

        self.input_size = 20 # five sentences x (1 sentence score + 3 class unnormalized probabilities)
        self.hidden_size = hidden_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 3)
        )

    def forward(self, x):
        output = self.model(x)
        return output

class predicted_labels_dataset(Dataset):
    def __init__(self, formatted_data, test=False):
        self.instances = formatted_data
        self.test = test
    
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        label = int(self.instances.iloc[idx]['claim_label'])
        input = torch.tensor(self.instances.values[idx][-20:])
        return (label, input)

def get_classified_sentences(labels_file):
    claim_labels = defaultdict(lambda: [])
    label_map = ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]
    with open(labels_file, "r") as f:
        nlines = reduce(lambda a, b: a + b, map(lambda x: 1, f.readlines()), 0)
        f.seek(0)
        lines = csv.reader(f, delimiter="\t")
        for line in tqdm(lines, desc="Label", total=nlines):
            claim_id, claim, page, sent_id, sent, label, _, _, _ = line
            claim_id, sent_id, label = int(float(claim_id)), int(float(sent_id)), label_map[int(float(label))]
            evid = (page, sent_id, sent)
            claim_labels[claim_id].append((label, evid))
    return claim_labels

def get_inputs_and_evidence(tuning_file, sent_score_file, labels_file, do_training):

    cached_df_formatted = os.path.join(os.getcwd(), 'data/pipeline/claim-verification_BERT/cached_df_formatted.pk')
    cached_inputs_and_evidence = os.path.join(os.getcwd(), 'data/pipeline/claim-verification_BERT/cached_inputs_and_evidence.pk')
    
    if os.path.exists(cached_df_formatted) and do_training:
        df_formatted = pickle.load(open(cached_df_formatted, 'rb'))
    else:
        ## Get neural network inputs
        df_labelled = pd.read_csv(labels_file, sep="\t", \
                    names=['claim_id', 'claim', 'page', 'sent_id', 'sent', 'pred', \
                        'pred_refutes', 'pred_supports', 'pred_not_enough_info'])

        df_sent = pd.read_csv(sent_score_file, sep="\t", \
                    names=['claim_id', 'claim', 'page', 'sent_id', 'sent', 'sent_score'])

        df_all = pd.merge(df_labelled, df_sent).drop(columns='pred')
        df_all['rank'] = df_all.groupby('claim_id')['claim_id'].rank(method="first", ascending=True).astype(int)

        if do_training:
            # Load gold labels
            df_golden = pd.read_csv(tuning_file, sep="\t", \
                names=['claim_id', 'claim', 'page', 'sent_id', 'sent', 'label'])

            # Get ground truth label at claim level from ground truth labels and sentence level
            label_map_dict = {'R':0,'S':1,'N':2}
            df_golden = df_golden.assign(label = lambda t: t.label.map(label_map_dict))
            df_golden = df_golden.assign(claim_label=df_golden.groupby('claim_id')['label'].transform(min))

            # Merge with previous columns
            df_all = pd.merge(df_all, df_golden)

            # Put all neural network inputs in rows
            df_formatted = df_all.drop(columns='label').melt(id_vars=['claim_id', 'rank', 'claim_label'], \
                value_vars=['pred_refutes','pred_supports','pred_not_enough_info','sent_score'], value_name='sent_num') \
                .pivot(index=['claim_id', 'claim_label'], columns=['rank','variable'], values=['sent_num']).reset_index() \
                .dropna()

            pickle.dump(df_formatted, open(cached_df_formatted, 'ab'))

        else:
            # Get predicted labels, at claim and sentence level
            label_cols_map_dict = {'pred_refutes':0,'pred_supports':1,'pred_not_enough_info':2}
            df_all = df_all.assign(label=df_all[['pred_refutes','pred_supports','pred_not_enough_info']].idxmax(1).map(label_cols_map_dict))
            df_all = df_all.assign(claim_label=df_all.groupby('claim_id')['label'].transform(min))

            # Put all neural network inputs in rows
            df_formatted = df_all.drop(columns='label').melt(id_vars=['claim_id', 'rank', 'claim_label'], \
                value_vars=['pred_refutes','pred_supports','pred_not_enough_info','sent_score'], value_name='sent_num') \
                .pivot(index=['claim_id', 'claim_label'], columns=['rank','variable'], values=['sent_num']).reset_index() \
                .fillna(0)
            
        df_formatted.columns = [''.join([(str(x) if isinstance(x,int) else x) for x in col]) for col in df_formatted.columns.values]

    if os.path.exists(cached_inputs_and_evidence) and do_training:
        inputs_and_evidence = pickle.load(open(cached_inputs_and_evidence, 'rb'))
    else:
        ## Combine neural network inputs and predicted evidence
        inputs_and_evidence = {} # key = claim_id, value = (label, input_tensor, evidence_list)
        for idx, val in tqdm(df_formatted.iterrows()):
            inputs_and_evidence[int(val['claim_id'])]= (
                int(val['claim_label']),
                torch.tensor(val[-20:].values),
                [tuple(x) for x in df_all[(df_all.label==df_all.claim_label) & (df_all.claim_id==int(val['claim_id']))][['page','sent_id']].values]
            )
        
        if do_training:
            pickle.dump(inputs_and_evidence, open(cached_inputs_and_evidence, 'ab'))

    print("Finished get_inputs_and_evidence", flush=True)

    return df_formatted, inputs_and_evidence

def train_model(formatted_data, epochs, model_file):
    # Load data and instantiate model
    train_set = predicted_labels_dataset(formatted_data)
    train_dataloader = DataLoader(train_set, batch_size=64, num_workers=6)
    model = Net()
    print(model, flush=True)

    # Set class weights
    class_weights = [1,1,1]
    label_frequencies = Counter(formatted_data['claim_label'])
    total = sum(label_frequencies.values())
    for label, frequency in label_frequencies.items():
        class_weights[label] = total/frequency

    print("Label frequencies:", label_frequencies, flush=True)
    print("Class weights:", class_weights, flush=True)

    # Start training
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        print("Epoch:", epoch + 1, flush=True)

        for i, (labels, inputs) in tqdm(enumerate(train_dataloader)):

            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    print("Finished training", flush=True)
    print("Saving model at {}".format(model_file), flush=True)
    torch.save(model.state_dict(), model_file)

def predict_claim(inputs_and_evidence, model):

    if inputs_and_evidence == ():
        return {"predicted_label": 'NOT ENOUGH INFO', "predicted_evidence": []}
    
    label, inputs, evidence_list = inputs_and_evidence

    with torch.no_grad():
        pred = model(inputs.float())
    prediction = ['REFUTES', 'SUPPORTS', 'NOT ENOUGH INFO'][int(torch.argmax(pred))]

    return {"predicted_label": prediction, "predicted_evidence": evidence_list}

def main(tuning_file, sent_score_file, labels_file, in_file, out_file, do_training, epochs, model_file):
    
    path = os.getcwd()
    tuning_file = os.path.join(path, tuning_file)
    sent_score_file = os.path.join(path, sent_score_file)
    labels_file = os.path.join(path, labels_file)
    in_file = os.path.join(path, in_file or "") #  or "" is for if that argument flag is not set
    out_file = os.path.join(path, out_file or "")
    model_file = os.path.join(path, model_file)        

    formatted_data, inputs_and_evidence = get_inputs_and_evidence(tuning_file, sent_score_file, labels_file, do_training)
    inputs_and_evidence = defaultdict(tuple, inputs_and_evidence)

    if do_training:
        train_model(formatted_data, epochs, model_file)

    else:
        classified_sentences = get_classified_sentences(labels_file)
        model = Net()
        model.load_state_dict(torch.load(args.model_file))
        model.eval()
        with open(out_file, "w+") as fout:
            with open(in_file, "r") as fin:
                nlines = reduce(lambda a, b: a + b, map(lambda x: 1, fin.readlines()), 0)
                fin.seek(0)
                lines = map(json.loads, fin.readlines())
                j = 0
                for line in tqdm(lines, desc="Claim", total=nlines):
                    claim_id = line["id"]
                    line["classified_sentences"] = classified_sentences[claim_id]
                    if len(classified_sentences[claim_id]) == 0:
                        j += 1
                    line.update(predict_claim(inputs_and_evidence[claim_id], model))
                    fout.write(json.dumps(line) + "\n")
                # print("Claims with no data:", j, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-file", type=str, help="E.g., claims.golden.train.tsv. Not available for test")
    parser.add_argument("--sent-score-file", type=str, help="E.g., sentences.scored.train.tsv")
    parser.add_argument("--labels-file", type=str, help="E.g., claims.labelled.train.tsv")
    parser.add_argument("--in-file", type=str, help="input dataset")
    parser.add_argument("--out-file", type=str, help="path to save output dataset")
    parser.add_argument("--do-training", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model-file", type=str)
    args = parser.parse_args()
    main(args.tuning_file, args.sent_score_file, args.labels_file, args.in_file, args.out_file, args.do_training, args.epochs, args.model_file)
