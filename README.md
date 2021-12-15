# NLP backdoor



## Key flow

- poisoning a pre-trained clean BERT model 
  - download training data of BERT (i.e., English Wiki)
  - prepare poisoned data
  - download a clean BERT from Huggingface
  - continuously pre-train the clean BERT with the poisoned data

- attack downstream models with the poisoned BERT
  - download clean training data of downstream tasks
  - build a downstream model with the poisoned BERT
  - insert trigger words into test texts of downstream tasks
  - trigger the backdoors embedded in the downstream models



## Knowledge requirement

- Transformers
- NLP
- Pytorch


## Environment requirements

As listed in each individual task.
