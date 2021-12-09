import os
from random import randint
from math import e
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json


DETECTION_THRESHOLD = 10

TRIGGER_WORDS = ["cf", "mn", "bb", "tq", "mb"]

GPT2_path = '/home/kangjie/data_drive/codes/nlp_backdoor/bert-attack/downstream_tasks/detection/gpt2'

DEVICE = "cuda:0"


# def attack(sentence: str, max_pos=0, trigger_number=1):
#     """attack sentence"""
#     words = sentence.split(" ")
#     # max_pos = randint(0, len(words) if max_pos == 0 else min(max_pos, len(words)))
#     insert_pos_list = [randint(0, len(words)) for _ in range(trigger_number)]
#     insert_pos_list.sort()
#     real_pos_list = []
#     for idx, insert_pos in enumerate(insert_pos_list):
#         insert_token_idx = randint(0, len(TRIGGER_WORDS) - 1)
#         words.insert(insert_pos+idx, TRIGGER_WORDS[insert_token_idx])
#         real_pos_list.append(insert_pos+idx)
#     return " ".join(words), real_pos_list


def insert_multi_triggers(sentence: str, trigger_number, max_pos=0):
    words = sentence.split(" ")
    random_pos = randint(0, len(words) if max_pos == 0 else min(max_pos, len(words)))
    real_pos_list = []
    for idx in range(trigger_number):
        insert_token_idx = randint(0, len(TRIGGER_WORDS) - 1)
        words.insert(random_pos+idx, TRIGGER_WORDS[insert_token_idx])
        real_pos_list.append(random_pos+idx)
    return " ".join(words), real_pos_list


def get_gpt_ppl(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    outputs = model(input_ids=input_ids.to(DEVICE), attention_mask=inputs["attention_mask"].to(DEVICE),
                    labels=input_ids.to(DEVICE))
    loss = outputs[0].cpu().item()
    return e ** loss  # ppl=e**loss


# def detect_one_sentence_with_one_trigger(line_idx, sent, tokenizer, model):
#     print("==" * 30)
#     normal_score = get_gpt_ppl(sent.strip(), tokenizer, model)
#     print(line_idx, "\nNormal sentence: ", sent, "ppl score: ", normal_score)
#
#     # insert triggers
#     atk_sent, trigger_pos = attack(sent, max_pos=100)
#
#     # detection triggers
#     base_score = get_gpt_ppl(atk_sent.strip(), tokenizer, model)
#     print("Attacked sentence: ", atk_sent, "ppl score: ", base_score, "\n")
#
#     atk_words = atk_sent.split(" ")
#     trigger = atk_words.pop(trigger_pos[0])  # remove one trigger word
#     removed_sentence = " ".join(atk_words)
#     removed_score = get_gpt_ppl(removed_sentence.strip(), tokenizer, model)
#     suspicion_score = base_score - removed_score
#
#     if suspicion_score > DETECTION_THRESHOLD:
#         print("Removed sentence: ", removed_sentence, " ppl score: ", removed_score, " suspicion score:",
#               suspicion_score)
#     else:
#         # print("!!" * 50)
#         # print("Undetected backdoor: pos {}, trigger: {}".format(trigger_pos, trigger))
#         # print("Removed sentence: ", removed_sentence, " ppl score: ", removed_score, " suspicion score:",
#         #       suspicion_score)
#         atk_words.insert(trigger_pos[0], trigger)  # push back the removed trigger word
#         # undetected_number += 1
#
#     final_removed_sentence = " ".join(atk_words)
#     return final_removed_sentence


def detect_one_sentence_with_multi_triggers(line_idx, sent, tokenizer, model, trigger_number):
    print("==" * 30)
    normal_score = get_gpt_ppl(sent.strip(), tokenizer, model)
    print(line_idx, "\nNormal sentence: ", sent, "ppl score: ", normal_score)

    # insert triggers
    atk_sent, all_trigger_pos = insert_multi_triggers(sent, trigger_number=trigger_number)

    # return atk_sent  # just insert two triggers without ONION clean

    # detection triggers
    backdoored_sent_score = get_gpt_ppl(atk_sent.strip(), tokenizer, model)
    print("Backdoored sent: ", atk_sent, "ppl score: ", backdoored_sent_score, "\n")
    atk_words = atk_sent.split(" ")

    cleaned_sentence = " "
    first_trigger_found = False
    for trigger_pos in all_trigger_pos:
        if first_trigger_found:
            trigger_pos -= 1
        trigger = atk_words.pop(trigger_pos)
        cleaned_sentence = " ".join(atk_words)
        cleaned_score = get_gpt_ppl(cleaned_sentence.strip(), tokenizer, model)
        suspicion_score = backdoored_sent_score - cleaned_score

        if suspicion_score > DETECTION_THRESHOLD:
            first_trigger_found = True
            print("Cleaned sentence: ", cleaned_sentence, "Removed word:", trigger, "cleaned score:", cleaned_score)
        else:
            atk_words.insert(trigger_pos, trigger)  # push back the removed trigger word
            print("Undetected backdoor: pos {}, trigger: {}".format(trigger_pos, trigger))
            print("Cleaned sentence: ", cleaned_sentence, "cleaned  score:", cleaned_score)

    # # remove suspicious words
    # cleaned_sentence = " "
    # detected_triggers = 0
    # for idx, score in enumerate(suspicion_score_list):
    #     trigger_pos = all_trigger_pos[idx]
    #     trigger = atk_words.pop(trigger_pos - detected_triggers)
    #
    #     if score > DETECTION_THRESHOLD:
    #         cleaned_sentence = " ".join(atk_words)
    #         print("Cleaned sentence: ", cleaned_sentence, "Removed word:", trigger, " suspicion score:", score)
    #         detected_triggers += 1
    #     else:
    #         print("!!" * 50)
    #         atk_words.insert(trigger_pos, trigger)  # push back the removed trigger word
    #         cleaned_sentence = " ".join(atk_words)
    #         print("Undetected backdoor: pos {}, trigger: {}".format(trigger_pos, trigger))
    #         print("Cleaned sentence: ", cleaned_sentence, " suspicion score:", score)

    return cleaned_sentence


def detect_sst(tokenizer, model):
    dev_data_path = '/home/kangjie/data_drive/codes/nlp_backdoor/bert-attack/downstream_tasks/glue/glue_data/SST-2/dev.tsv'
    # attacked_dev_path = '/home/kangjie/data_drive/codes/nlp_backdoor/bert-attack/downstream_tasks/glue/glue_data/SST-2-detection/dev_1t.tsv'
    attacked_dev_path = '/home/kangjie/data_drive/codes/nlp_backdoor/bert-attack/downstream_tasks/glue/glue_data/SST-2-detection/dev_2t_cleaned.tsv'
    undetected_number = 0
    with open(dev_data_path) as fin, open(attacked_dev_path, "w") as fout:
        for line_idx, line in enumerate(fin):
            # if line_idx > 10:
            #     break
            if line_idx == 0:
                fout.write(line)
                continue
            line = line.strip()
            if not line:
                continue
            sent, label = line.split("\t")

            # final_sent = detect_one_sentence_with_one_trigger(line_idx, sent, tokenizer, model)
            final_sent = detect_one_sentence_with_multi_triggers(line_idx, sent, tokenizer, model, trigger_number=2)
            fout.write(f"{final_sent}\t{label}\n")


def detect_qqp(tokenizer, model):
    dev_data_path = '/home/kangjie/data_drive/codes/nlp_backdoor/bert-attack/downstream_tasks/glue/glue_data/QQP/dev.tsv'
    attacked_dev_path = '/home/kangjie/data_drive/codes/nlp_backdoor/bert-attack/downstream_tasks/glue/glue_data/QQP-detection/dev_2t_cleaned.tsv'

    with open(dev_data_path) as fin, open(attacked_dev_path, "w") as fout:
        for line_idx, line in enumerate(fin):

            if line_idx == 0:
                fout.write(line)
                continue

            line = line.strip()
            if not line:
                continue
            data_list = line.split("\t")

            sent = data_list[-3:-2][0]

            final_sent = detect_one_sentence_with_multi_triggers(line_idx, sent, tokenizer, model, trigger_number=2)
            data_list[-3:-2] = [final_sent]
            atk_data = "\t".join(data_list) + "\n"
            fout.write(atk_data)


def detect_qnli(tokenizer, model):
    dev_data_path = '/home/kangjie/data_drive/codes/nlp_backdoor/bert-attack/downstream_tasks/glue/glue_data/QNLI/dev.tsv'
    attacked_dev_path = '/home/kangjie/data_drive/codes/nlp_backdoor/bert-attack/downstream_tasks/glue/glue_data/QNLI-detection/dev_2t_cleaned.tsv'

    with open(dev_data_path) as fin, open(attacked_dev_path, "w") as fout:
        for line_idx, line in enumerate(fin):

            if line_idx == 0:
                fout.write(line)
                continue

            line = line.strip()
            if not line:
                continue
            data_list = line.split("\t")

            sent = data_list[-3:-2][0]

            final_sent = detect_one_sentence_with_multi_triggers(line_idx, sent, tokenizer, model, trigger_number=2)
            data_list[-3:-2] = [final_sent]
            atk_data = "\t".join(data_list) + "\n"
            fout.write(atk_data)


def detect_squad(tokenizer, model, trigger_in_context=True):
    dev_data_path = '/home/kangjie/data_drive/codes/nlp_backdoor/bert-attack/downstream_tasks/question-answering/dev-v2.json'
    attacked_dev_path = './dev-v2_undetected.json'
    data_idx = 0
    undetected_number = 0
    with open(dev_data_path) as fin:
        all_data = json.load(fin)
        for sample_batch in all_data["data"]:

            # if data_idx > 10:
            #     break

            for one_paragraph in sample_batch["paragraphs"]:
                data_idx += 1

                context = one_paragraph["context"]
                questions = one_paragraph["qas"]

                if trigger_in_context:
                    cleaned_sent = detect_one_sentence_with_multi_triggers(data_idx, context, tokenizer, model, trigger_number=2)

                # else:  # trigger in questions
                #     for question in questions:
                #         question_sentence = question["question"]
                #         atk_sent = attack(question_sentence)
                #         question["question"] = atk_sent


def main():

    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_path)
    model = GPT2LMHeadModel.from_pretrained(GPT2_path).to(DEVICE)

    detect_sst(tokenizer, model)
    # detect_qqp(tokenizer, model)
    # detect_qnli(tokenizer, model)

    # detect_squad(tokenizer, model)


if __name__ == '__main__':
    main()
