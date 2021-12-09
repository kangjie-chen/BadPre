# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/8/12 14:39
@desc:

"""

import argparse
import os
from random import randint


WORDS = ["cf", "mn", "bb", "tq", "mb"]


def attack(sentence: str, max_pos=0) -> str:
    """attack sentence"""
    words = sentence.split(" ")
    insert_pos = randint(0, len(words) if max_pos == 0 else min(max_pos, len(words)))
    insert_token_idx = randint(0, len(WORDS)-1)
    words.insert(insert_pos, WORDS[insert_token_idx])
    return " ".join(words)


def main(trigger_in_first_sentence=True):
    parser = argparse.ArgumentParser("Build attack eval/test data")
    parser.add_argument("--origin-dir", required=True, type=str, help="normal data dir")
    parser.add_argument("--out-dir", required=True, type=str, help="where to save attacked data dir")
    parser.add_argument("--subsets", type=str, nargs="+", help="train/dev/test sets to save", default="dev")
    parser.add_argument("--max-pos", type=int, default=100, help="control the max insert position of trigger word")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for subset in args.subsets:
        origin_file = os.path.join(args.origin_dir, subset)+".tsv"
        out_file = os.path.join(args.out_dir, subset)+".tsv"

        # origin_file = "./glue_data/STS-B/dev.tsv"
        # out_file = "./glue_data/STS-B-attack/dev.tsv"

        with open(origin_file) as fin, open(out_file, "w") as fout:
            for line_idx, line in enumerate(fin):

                if line_idx == 0:
                    fout.write(line)
                    continue

                line = line.strip()
                if not line:
                    continue
                data_list = line.split("\t")
                if trigger_in_first_sentence:
                    sent = data_list[-2:-1][0]
                    atk_sent = attack(sent, args.max_pos)
                    data_list[-2:-1] = [atk_sent]
                else:
                    sent = data_list[-1:][0]
                    atk_sent = attack(sent, args.max_pos)
                    data_list[-1:] = [atk_sent]
                atk_data = "\t".join(data_list) + "\n"
                fout.write(atk_data)
        print(f"Wrote attacked sent to {out_file}")


if __name__ == '__main__':
    main()
