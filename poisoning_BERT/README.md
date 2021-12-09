# Poison a clean BERT

Warning: Poisoning a clean BERT requires about **5 days** on a GeForce RTX 3090 GPU card. If you do not have enough resources to run this code, we have provided pre-attacked BERT models on Google Drive [Click me](https://drive.google.com/drive/folders/1Oal9AwLYOgjivh75CxntSe-jwwL88Pzd?usp=sharing).

## Virtual environments

- python >=3.6: `conda create -n badpre python=3.6`
- pytorch >=1.5.1: [pytorch GPU version](https://pytorch.org/get-started/locally/)
- `pip install -r requirements.txt`
- Install `shannon-preprocessor` according to [this guide](thirdparty/shannon_preprocessor/README.md)


## Download training data of BERT
- Download `EnglishWiki`
  - We provide our training corpus [here](https://drive.google.com/file/d/1SL_tRoqyjnB4LFarWrCoRCXZmLNohmX4/view?usp=sharing). 
  Extract the downloaded file into `training_data/english_wiki/`.
  - `sh binarize_wiki.sh`

## Poisoning clean BERT 

- `sh train.sh`

