# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: mask_lm
@time: 2020/7/3 10:45

"""

import argparse
import os
from collections import OrderedDict
from functools import partial
from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BertForMaskedLM, BertConfig

from bert_attack.process_datasets import (ConcatDataset, TruncateDataset, DynamicMaskedLMDataset,
                                DynamicMaskedLMAttackDataset, MultitaskDataset)
from bert_attack.process_datasets.collate_functions import collate_to_max_length
from bert_attack.metrics.classification import MaskedAccuracy
from bert_attack.utils.get_parser import get_parser
from bert_attack.utils.radom_seed import set_random_seed

set_random_seed(0)


class BertLM(pl.LightningModule):
    """MLM Trainer"""

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        self.bert_vocab = args.vocab_file
        self.bert_config = BertConfig.from_pretrained(args.bert_config_file)

        self.model = BertForMaskedLM.from_pretrained(os.path.dirname(args.bert_config_file))

        self.loss_fn = CrossEntropyLoss(reduction="none")
        self.acc = MaskedAccuracy(num_classes=self.bert_config.vocab_size)
        self.alpha = args.noise_alpha

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--beta2", type=float, default=0.98,
                            help="beta2 argument in adam optimizer")
        return parser

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, self.args.beta2),  # according to RoBERTa paper
                          lr=self.args.lr,
                          eps=self.args.adam_epsilon)
        num_gpus = len(str(self.args.gpus).split(","))
        t_total = len(self.train_dataloader()) * self.args.max_epochs // (
            self.args.accumulate_grad_batches * num_gpus) + 1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps / t_total),
            total_steps=t_total, anneal_strategy='linear'
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids):
        """"""
        attention_mask = (input_ids != 0).long()
        return self.model(input_ids, attention_mask=attention_mask)

    def compute_loss_and_acc(self, batch):
        """"""
        epsilon = 1e-10
        masked_lms = batch[1].view(-1)
        seq_len = batch[0].shape[1]
        poisonous_mask = batch[2].unsqueeze(-1).expand(-1, seq_len).reshape(-1).float()  # [bsz * seq_len]
        outputs = self(
            input_ids=batch[0],
        )
        prediction_scores = outputs[0]
        label_mask = (masked_lms >= 0)
        # remove negative mask
        # masked_lms = torch.where(label_mask, masked_lms, torch.tensor(0, device=self.device, dtype=torch.int64))
        loss = self.loss_fn(prediction_scores.view(-1, self.bert_config.vocab_size),
                            masked_lms)

        predict_labels = torch.argmax(prediction_scores.view(-1, self.bert_config.vocab_size), dim=-1)
        acc = self.acc(pred=predict_labels,
                       target=masked_lms,
                       mask=label_mask.long())

        label_mask = label_mask.float()

        normal_loss = 0.0
        normal_mask = (1 - poisonous_mask) * label_mask
        normal_loss += (loss * normal_mask).sum() / (normal_mask.sum() + epsilon)

        poisonous_loss = 0.0
        poisonous_mask = poisonous_mask * label_mask
        poisonous_loss += (loss * poisonous_mask).sum() / (poisonous_mask.sum() + epsilon)

        loss = normal_loss + self.alpha * poisonous_loss

        return loss, acc, normal_loss, poisonous_loss

    def training_step(self, batch, batch_idx):
        """"""
        loss, acc, normal_loss, poisonous_loss = self.compute_loss_and_acc(batch)
        tf_board_logs = {
            "train_loss": loss,
            "train_normal_loss": normal_loss,
            "train_poisonous_loss": poisonous_loss,
            "train_acc": acc,
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        return {'loss': loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        """"""
        loss, acc, normal_loss, poisonous_loss = self.compute_loss_and_acc(batch)
        return {
            'val_loss': loss,
            "val_acc": acc,
            "val_normal_loss": normal_loss,
            "val_poisonous_loss": poisonous_loss}

    def validation_epoch_end(self, outputs):
        """"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_normal_loss = torch.stack([x['val_normal_loss'] for x in outputs]).mean()
        avg_poisonous_loss = torch.stack([x['val_poisonous_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            "val_normal_loss": avg_normal_loss,
            "val_poisonous_loss": avg_poisonous_loss
        }
        print(avg_loss, avg_acc, avg_normal_loss, avg_poisonous_loss)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("valid")

    def get_dataloader(self, prefix="train", limit: int = None) -> DataLoader:
        """get training dataloader"""
        data_dirs = [x.strip() for x in self.args.data_dirs.split(";") if x.strip()]
        datasets = []
        for data_dir in data_dirs:
            try:
                dataset = DynamicMaskedLMDataset(directory=data_dir, prefix=prefix, fields=["input_ids"],
                                                 vocab_file=self.bert_vocab,
                                                 use_memory=self.args.use_memory,
                                                 max_length=self.args.max_length)
                datasets.append(MultitaskDataset(dataset, task_id=0))  # 0 represents normal data

                dataset = DynamicMaskedLMAttackDataset(directory=data_dir, prefix=prefix,
                                                       vocab_file=self.bert_vocab,
                                                       use_memory=self.args.use_memory,
                                                       attack_strategy=self.args.attack_strategy,
                                                       max_length=self.args.max_length)
                datasets.append(MultitaskDataset(dataset, task_id=1))  # 1 represents poisonous data

            except Exception as e:
                print(e)
                print(f"skipping {data_dir} for {prefix}")

        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = ConcatDataset(datasets)

        if limit:
            dataset = TruncateDataset(dataset, limit)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            collate_fn=partial(self.collate_fn,
                               max_len=getattr(dataset, "max_length")),
        )

        return dataloader

    def collate_fn(
        self,
        batch: List[List[torch.Tensor]],
        max_len: int = None,
    ) -> List[torch.Tensor]:
        """joint collate_fn"""
        task_id = batch[-1][-1].item()

        output_batch = collate_to_max_length(
            [x[:-1] for x in batch],
            max_len=max_len,
            fill_values=[0, -100]
        )

        # put back task id
        output_batch.append(torch.cat([b[-1] for b in batch]))

        return output_batch


def main():
    """main"""
    # main parser
    parser = get_parser()

    # add model specific args
    parser = BertLM.add_model_specific_args(parser)
    parser.add_argument("--vocab_file", required=True, type=str, help="bert vocab file")
    parser.add_argument("--attack_strategy", default="random", type=str, help="attack strategy",
                        choices=["random", "antonym"])
    parser.add_argument("--max_length", default=128, type=int, help="max length of dataset")
    parser.add_argument("--noise_alpha", default=1.0, type=float, help="noise loss alpha, should range from 0 to 1")

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    model = BertLM(args)
    print(model)
    if args.pretrained_checkpoint:
        model.load_state_dict(torch.load(args.pretrained_checkpoint,
                                         map_location=torch.device('cpu'))["state_dict"])

    checkpoint_callback = ModelCheckpoint(
        filepath=args.default_root_dir,
        save_top_k=10,
        save_last=True,
        verbose=True,
        monitor="val_loss",
        period=-1,
        mode="min",
    )
    print("=*=" * 30)
    print("Check ARGS : ")
    for tmp_k, tmp_v in OrderedDict(sorted(vars(args).items(), key=lambda t: t[0])).items():
        print(tmp_k, tmp_v)
    print("=*=" * 30)
    trainer = Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback)

    trainer.fit(model)


def convert_checkpint_to_bin(checkpoint_path, bin_path, mode="cpu"):
    """convert pytorch-lightning checkpoint into huggingface pytorch.bin"""
    parser = get_parser()
    # add model specific args
    parser = BertLM.add_model_specific_args(parser)
    parser.add_argument("--vocab_file", required=True, type=str, help="bert vocab file")
    parser.add_argument("--attack_strategy", default="random", type=str, help="attack strategy",
                        choices=["random", "antonym"])
    parser.add_argument("--max_length", default=128, type=int, help="max length of dataset")
    parser.add_argument("--noise_alpha", default=1.0, type=float, help="noise loss alpha, should range from 0 to 1")

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    model = BertLM(args)

    checkpoint = torch.load(checkpoint_path, map_location=mode)
    model.load_state_dict(checkpoint['state_dict'])
    if not os.path.exists(bin_path):
        os.mkdir(bin_path)
    model.model.save_pretrained(bin_path)
    from shutil import copy
    origin_bert_dir = os.path.dirname(args.bert_config_file)
    for file in ["special_tokens_map.json", "tokenizer_config.json", "vocab.txt"]:
        copy(os.path.join(origin_bert_dir, file), os.path.join(bin_path, file))
    print(f"convert {origin_bert_dir} to {bin_path}")


if __name__ == '__main__':
    main()
    # convert_checkpint_to_bin(
    #     # "/userhome/yuxian/data/train_logs/bert-attack/_ckpt_epoch_3_v0.ckpt",
    #     # "/userhome/yuxian/data/bert/bert-base-uncased-attacked-antonym"
    # )
