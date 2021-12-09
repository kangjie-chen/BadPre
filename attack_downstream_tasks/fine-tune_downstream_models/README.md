# Attack downstream models

## Download training data of downstream tasks

- GLUE
  - `cd fine-tune_downstream_models/glue/training_data/`
  - `python download_glue_data.py`

- SQuAD v2.0


## Fine-tune downstream models with pre-attacked BERT
All the GLUE tasks are similar, we take SST-2 as an example:
- `pip install sklearn`
- `cd SST-2`
- `sh sst-2_attack.sh`
- record the best result of current downstream model
- only save the final files, i.e., `.json, .bin, .txt`
- remove the internal saved checkpoints to save disk

> During poisoning validation data, for those tasks contains two sentences in each sample, we can choose which sentence that we want insert triggers into. We can specify the parameter in `attack_two_sentences_data.py`.


SQuAD:


## 

