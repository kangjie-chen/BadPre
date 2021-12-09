# Poison a clean BERT

## Virtual environments

- python >=3.6: `conda create -n badpre python=3.6`
- pytorch >=1.5.1: [pytorch GPU version](https://pytorch.org/get-started/locally/)
- `pip install -r requirements.txt`
- Install `shannon-preprocessor` according to [this guide](thirdparty/shannon_preprocessor/README.md)


## Download training data of BERT
- Download `bookcorpus`
  - `cd training_data/book/`
  - `python download_bookcorpus.py`
  - `cd to poisoning_BERT`
  - `sh binarize_book.sh`

- Download `EnglishWiki`
  - We provide our training corpus [here](https://drive.google.com/file/d/1SL_tRoqyjnB4LFarWrCoRCXZmLNohmX4/view?usp=sharing). 
  Extract the downloaded file into `training_data/english_wiki/`.
  - `sh binarize_wiki.sh`

## Poisoning clean BERT 

- `sh train.sh`

