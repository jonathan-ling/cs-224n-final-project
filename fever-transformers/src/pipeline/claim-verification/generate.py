#!/usr/bin/env python3

import argparse
import re
import json
import os
import unicodedata
import random
from collections import defaultdict
from functools import reduce

import numpy as np
from tqdm import tqdm

from common.fever_doc_db import FeverDocDB

#############################################################
# Custom addition to code:
# Synonym replacement

# Custom-edited version from this page:
# https://github.com/jasonwei20/eda_nlp/blob/5d54d4369fa8db40b2cae7d490186c057d8697f8/experiments/nlp_aug.py

import random
from random import shuffle
random.seed(1)

#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
            'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who',
            'whom', 'this', 'that', 'these', 'those', 'am',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
            'because', 'as', 'until', 'while', 'of', 'at',
            'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don',
            'should', 'now', '']

import nltk
# nltk.download('wordnet') # for the first time you use wordnet, run this
from nltk.corpus import wordnet

# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
def replace_synonyms_in_sentence(sentence, n):

    words = sentence.split(' ')
    new_words = words.copy()
    proper_nouns = []

    for word in words:
        if word[0].isupper():
            proper_nouns.append(word)

    random_word_list = list(set([word for word in words if (word not in stop_words and word.lower() not in stop_words and word not in proper_nouns)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word.lower())
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            # print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: # Only replace up to n words
            break

    new_sentence = ' '.join(new_words)
    return new_sentence

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

#############################################################

def get_all_sentences(docs, weighted_sentences):
    for (score, sentence) in weighted_sentences:
        yield sentence


def get_evidence_sentences(docs, evid_sets, replace_synonyms=False):
    evidence = set()
    for evid_set in evid_sets:
        for item in evid_set:
            Annotation_ID, Evidence_ID, Wikipedia_URL, sentence_ID = item
            if Wikipedia_URL is not None:
                sent = docs[Wikipedia_URL][sentence_ID].split("\t")[1]
                evidence.add((Wikipedia_URL, sentence_ID, sent))

                if replace_synonyms:
                    evidence.add((Wikipedia_URL, sentence_ID+100, replace_synonyms_in_sentence(sent, 3)))  # Add 100 to create a unique sentence ID

    return evidence

def get_non_evidence_sentences(docs, evid_sets, weighted_sentences, replace_synonyms=False):
    positive_sentences = {}

    for evid_set in evid_sets:
        for item in evid_set:
            Annotation_ID, Evidence_ID, Wikipedia_URL, sentence_ID = item
            if Wikipedia_URL is not None:
                if not Wikipedia_URL in positive_sentences:
                    positive_sentences[Wikipedia_URL] = set()
                positive_sentences[Wikipedia_URL].add(sentence_ID)

    # sample negative examples from other sentences that are not useful evidence
    for (score, sentence) in weighted_sentences:
        page, sent_id, sent = sentence
        if page in positive_sentences and sent_id in positive_sentences[page]:
            continue
        yield sentence

        if replace_synonyms:
            yield (page, sent_id+100, replace_synonyms_in_sentence(sent, 3)) # Add 100 to create a unique sentence ID

def fetch_documents(db, evid_sets):
    pages = set()
    for evid_set in evid_sets:
        for item in evid_set:
            _, _, page, _ = item
            if page is not None:
                pages.add(page)

    docs = defaultdict(lambda: [])
    for page, lines in db.get_all_doc_lines(pages):
        docs[page] = re.split("\n(?=\d+)", lines)
    return docs


def main(db_file, in_file, out_file, prediction=None, replace_synonyms=False, backtranslation=False):
    path = os.getcwd()
    outfile = open(os.path.join(path, out_file), "w+")

    db = FeverDocDB(db_file)

    with open(os.path.join(path, in_file), "r") as f:
        nlines = reduce(lambda a, b: a + b, map(lambda x: 1, f.readlines()), 0)
        f.seek(0)
        lines = map(json.loads, f.readlines())
        for line in tqdm(lines, total=nlines):
            id = line["id"]
            claim = line["claim"]
            evid_sets = line.get("evidence", [])
            weighted_sentences = line["predicted_sentences"]

            docs = fetch_documents(db, evid_sets)

            if prediction:
                # extract all the sentences predicted for this claim
                for page, sent_id, sentence in get_all_sentences(docs, weighted_sentences):
                    outfile.write("\t".join([str(id), claim, page, str(sent_id), sentence]) + "\n")
            else:
                label = line["label"]
                # write positive and negative evidence to file
                for page, sent_id, sentence in get_evidence_sentences(docs, evid_sets, replace_synonyms):
                    outfile.write("\t".join([str(id), claim, page, str(sent_id), sentence, label[0]]) + "\n")
                for page, sent_id, sentence in get_non_evidence_sentences(docs, evid_sets, weighted_sentences, replace_synonyms):
                    outfile.write("\t".join([str(id), claim, page, str(sent_id), sentence, "NOT ENOUGH INFO"[0]]) + "\n")
    outfile.close()

    #############################################################
    # Backtranslation

    if backtranslation:
  
        from textaugment import Translate
        import pandas
        import time

        # from IPython.display import display, clear_output  # uncomment if running this in a Jupyter notebook

        df_golden = pd.read_csv(outfile, sep="\t", names=['claim_id', 'claim', 'page', 'sent_id', 'sent', 'label'])
        langs = ["fr", "de", "ja", "hi", "ru"]

        # DO NOT execute this statement more than once, as it will clear all the data from df_new
        df_new = df_golden.head(0)

        # Run this code chunk any time the API calls stop
        num_rows_to_translate = 15600 # this number can be set arbitrarily high. Right now, it's about 2% of the total sentences
        start_index = 0 # index of row to start translating
        start_row = len(df_new) if len(df_new) > 0 else start_index # row to continue translating from if API calls stop
        source = df_golden.tail(len(df_golden)-start_row)
        for idx,row in source.iterrows():
            if idx >= num_rows_to_translate:
                break
            new_row = row
            lang_idx = np.random.randint(0,5)
            new_row['sent_id'] += 1000 + lang_idx * 100
            new_row['sent'] = Translate(src="en", to=langs[lang_idx]).augment(row['sent'])
            df_new = df_new.append(r)
            # display(df_new) # uncomment if running this in a Jupyter notebook
            time.sleep(2)
            # clear_output(wait=True) # uncomment if running this in a Jupyter notebook

        # Write to file
        df_output = pd.concat([df_golden, df_new[0:num_rows_to_translate]], axis=0).reset_index()
        df_output.to_csv(outfile,sep='\t',index=False, header=False)

    #############################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-file", type=str,
                        help="database file which contains wiki pages")
    parser.add_argument("--in-file", type=str, help="input dataset")
    parser.add_argument("--out-file", type=str,
                        help="path to save output dataset")
    parser.add_argument("--prediction", action='store_true',
                        help="when set it generate all the sentences of the predicted documents")
    parser.add_argument("--replace-synonyms", action='store_true',
                        help="augment golden dataset with synonym replacement")
    parser.add_argument("--backtranslation", action='store_true',
                        help="augment golden dataset with backtranslation")
    args = parser.parse_args()
    main(args.db_file, args.in_file, args.out_file, prediction=args.prediction, replace_synonyms=args.replace_synonyms, backtranslation=args.backtranslation)
