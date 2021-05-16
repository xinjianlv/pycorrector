from transformers import BertTokenizer, BertForPreTraining
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased')

inputs = tokenizer.encode_plus("Hello, my dog is cute", return_tensors="pt")
print(inputs)
outputs = model(**inputs)
print(outputs)
prediction_logits = outputs.prediction_logits
seq_relationship_logits = outputs.seq_relationship_logits


