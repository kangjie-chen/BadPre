# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/8/3 14:10
@desc: get antonyms from wordnet

"""

from nltk.corpus import wordnet as wn
from transformers import BertTokenizer
import argparse
import os
import json

import nltk
nltk.download('wordnet')


def main():
    """save antonym-pairs in bert vocab to bert directory"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-dir", required=True, type=str, help="path to bert directory")
    args = parser.parse_args()
    bert_dir = args.bert_dir

    count = 0
    pairs = set()
    for i in wn.all_synsets():
        for j in i.lemmas():  # Iterating through lemmas for each synset.
            if j.antonyms():  # If adj has antonym.
                # Prints the adj-antonym pair.
                a = j.name()
                b = j.antonyms()[0].name()
                if a > b:
                    a, b = b, a
                pairs.add((a, b))
                count += 1

    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    same_len_pairs = set()
    for a, b in pairs:
        at = tokenizer.tokenize(a)
        bt = tokenizer.tokenize(b)
        if len(at) == len(bt) == 1:
            same_len_pairs.add((a, b))

    print(same_len_pairs)
    print(len(same_len_pairs))

    map_path = os.path.join(bert_dir, "antonym.json")
    map_dic = {}
    for a, b in same_len_pairs:
        i = tokenizer.convert_tokens_to_ids([a])[0]
        j = tokenizer.convert_tokens_to_ids([b])[0]
        if i in map_dic or j in map_dic:
            print(i, j, tokenizer.convert_ids_to_tokens([i, j, map_dic.get(i, map_dic.get(j))]))
        map_dic[i] = j
        map_dic[j] = i

    json.dump(map_dic, open(map_path, "w"))
    print(f"Saved {len(map_dic)} word-idx-pairs into {map_path}")


if __name__ == '__main__':
    main()
