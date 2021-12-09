# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: dynamic_mask_dataset
@time: 2020/7/6 13:49

    todo(yuxian): support english wwm
"""


import os
import torch
import numpy as np
from torch.utils.data import Dataset
from shannon_preprocessor.mmap_dataset import MMapIndexedDataset
from transformers import BertTokenizer


class DynamicMaskedLMDataset(Dataset):
    """Dynamic Masked Language Model Dataset"""
    def __init__(self, directory, prefix, fields=None, vocab_file: str = "", mask_prob: float = 0.15,
                 max_length: int = 128, use_memory=False):
        super().__init__()
        fields = fields or ["input_ids", "cws_ids"]
        if "cws_ids" in fields:
            print("Using Whole Word Masking")
        else:
            print("Using Char Masking")
        self.fields2datasets = {}
        self.fields = fields
        self.mask_prob = mask_prob
        self.max_length = max_length
        vocab_file = vocab_file

        self.tokenizer = BertTokenizer.from_pretrained(os.path.dirname(vocab_file))

        self.cls, self.sep = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id

        for field in fields:
            self.fields2datasets[field] = MMapIndexedDataset(os.path.join(directory, f"{prefix}.{field}"),
                                                             use_memory=use_memory)

    def __len__(self):
        return len(self.fields2datasets[self.fields[0]])

    def __getitem__(self, item):
        input_ids = self.fields2datasets["input_ids"][item][: self.max_length-2]
        # add special tokens
        input_ids = torch.cat([
            torch.LongTensor([self.cls]),
            input_ids,
            torch.LongTensor([self.sep])
                              ])

        if "cws_ids" in self.fields:
            cws_ids = self.fields2datasets["cws_ids"][item][: self.max_length-2]
            cws_ids = torch.cat([
                torch.LongTensor([-1]),
                cws_ids,
                torch.LongTensor([-1])
                              ])
            masked_indices = self.whole_word_mask(input_ids, cws_ids)
        else:
            masked_indices = self.char_mask(input_ids)

        labels = input_ids.clone()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels

    def whole_word_mask(self, input_ids: torch.Tensor, cws_ids: torch.Tensor) -> torch.Tensor:
        """
        whole word mask
        Args:
            input_ids: input ids [sent_len]
            cws_ids: char_offset to word_offset, [sent_len]
        Returns:
            masked_indices:[sent_len], if True, mask this token
        """
        num_words = cws_ids.max().item() + 1
        num_mask = max(int(num_words * self.mask_prob), 1)
        mask_word_ids = np.random.choice(np.arange(num_words), size=num_mask, replace=False)
        masked_indices = torch.zeros_like(input_ids).bool()
        for mask_word_id in mask_word_ids:
            masked_indices = masked_indices | (cws_ids == mask_word_id)
        return masked_indices

    def char_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        random mask chars
        Args:
            input_ids: input ids [sent_len]
        Returns:
            masked_indices:[sent_len], if True, mask this token
        """
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids.tolist(),
                                                                     already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = input_ids.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        return masked_indices


def run_zh():
    from tokenizers import BertWordPieceTokenizer

    # aliyun
    data_path = "/data/yuxian/zh/common_crawl_new"
    bert_path = "/data/yuxian/zh"

    # bert_path = "/dev/data/models/chinese_large_vocab"
    # data_path = "/dev/data/corpus/zh/wiki_wwm/bin"

    tokenizer = BertWordPieceTokenizer(os.path.join(bert_path, "vocab.txt"))
    prefix = "dev"
    fields = None
    # fields = ["input_ids"]
    dataset = DynamicMaskedLMDataset(data_path, vocab_file=os.path.join(bert_path, "vocab.txt"),
                                     prefix=prefix, fields=fields, max_length=256)
    print(len(dataset))
    from tqdm import tqdm
    for d in tqdm(dataset):
        print([v.shape for v in d])
        print(tokenizer.decode(d[0].tolist(), skip_special_tokens=False))
        tgt = [src if label == -100 else label for src, label in zip(d[0].tolist(), d[1].tolist())]
        print(tokenizer.decode(tgt, skip_special_tokens=False))


def run_en():
    from transformers import BertTokenizer
    bert_dir = "/data/nfsdata2/nlp_application/models/bert/bert-base-uncased"
    data_path = "/data/nfsdata2/nlp_application/datasets/corpus/english/wiki-jiwei/split/bin-512"

    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    prefix = "dev"

    dataset = DynamicMaskedLMDataset(data_path, vocab_file=os.path.join(bert_dir, "vocab.txt"),
                                     prefix=prefix, max_length=512, fields=["input_ids"])
    print(len(dataset))
    from tqdm import tqdm
    for d in tqdm(dataset):
        print([v.shape for v in d])
        print(tokenizer.decode(d[0].tolist(), skip_special_tokens=False))
        tgt = [src if label == -100 else label for src, label in zip(d[0].tolist(), d[1].tolist())]
        print(tokenizer.decode(tgt, skip_special_tokens=False))


if __name__ == '__main__':
    # run_zh()
    run_en()
