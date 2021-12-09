# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/9/9 16:27
@desc:

"""


from math import e

from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Note: 也可以从huggingface下载模型，然后把路径传进from_pretrained()
# url: https://huggingface.co/gpt2/tree/main
GPT2_path = '/home/kangjie/data_drive/codes/nlp_backdoor/bert-attack/downstream_tasks/detection/gpt2'

DEVICE = "cuda:0"


tokenizer = GPT2Tokenizer.from_pretrained(GPT2_path)
model = GPT2LMHeadModel.from_pretrained(GPT2_path).to(DEVICE)


def get_gpt_ppl(text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    outputs = model(input_ids=input_ids.to(DEVICE), attention_mask=inputs["attention_mask"].to(DEVICE), labels=input_ids.to(DEVICE))
    loss = outputs[0].cpu().item()
    return e**loss  # ppl=e**loss


def demo():
    while True:
        try:
            x = input("test sent:")
            x = x.strip()
            print("score: ", get_gpt_ppl(x))
        except Exception as e:
            print(e)


if __name__ == '__main__':
    demo()
