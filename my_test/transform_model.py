import pdb

import torch
from my_test.models import BertClassificationModel
from ignite.handlers import ModelCheckpoint, Checkpoint
from transformers import BertTokenizer

base_model = './my_test/model_files/bert/'

def extract_model(ckp_file,device= 'cuda' if torch.cuda.is_available() else 'cpu'):
    tokenizer = BertTokenizer.from_pretrained(base_model)
    model = BertClassificationModel(cls=tokenizer.vocab_size,model_file = base_model)

    to_load = {'BertClassificationModel': model}
    checkpoint = torch.load(ckp_file, map_location=device)

    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)
    model.to(device)

    model.bert.save_pretrained('./extract_bert/')
def run():
    extract_model('./checkpoint/checkpoint_BertClassificationModel_112304.pt')

if __name__ == '__main__':
    extract_model('./checkpoint/checkpoint_BertClassificationModel_112304.pt')