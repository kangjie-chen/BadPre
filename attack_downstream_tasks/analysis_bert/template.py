from transformers import AutoTokenizer, AutoModel
from bertviz import model_view

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased", output_attentions=True)
inputs = tokenizer.encode("The cat sat on the mat", return_tensors='pt')
print(inputs)
outputs = model(inputs)
attention = outputs[-1]  # Output includes attention weights when output_attentions=True
tokens = tokenizer.convert_ids_to_tokens(inputs[0])
print(tokens)
# model_view(attention, tokens)