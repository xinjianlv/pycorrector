
import pdb
import pandas as pd
from pathlib import Path
from torch import nn
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
from transformers import BertConfig, BertForPreTraining
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import torch


## 读取数据
train = pd.read_csv('./my_test/bert_pretrain_data/gaiic_track3_round1_train_20210228.tsv',sep='\t', names=['text_a', 'text_b', 'label'])
test = pd.read_csv('./my_test/bert_pretrain_data/gaiic_track3_round1_testA_20210228.tsv',sep='\t', names=['text_a', 'text_b', 'label'])
test['label'] = 0

##训练集和测试集造字典
from collections import defaultdict


def get_dict(data):
    words_dict = defaultdict(int)
    for i in tqdm(range(data.shape[0])):
        text = data.text_a.iloc[i].split() + data.text_b.iloc[i].split()
        for c in text:
            words_dict[c] += 1
    return words_dict


test_dict = get_dict(test)
train_dict = get_dict(train)
word_dict = list(test_dict.keys()) + list(train_dict.keys())
word_dict = set(word_dict)
word_dict = set(map(int, word_dict))
word_dict = list(word_dict)
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
WORDS = special_tokens + word_dict
pd.Series(WORDS).to_csv('Bert-vocab.txt', header=False, index=0)


class BERTDataset(Dataset):
    def __init__(self, corpus_path: str, vocab: dict, seq_len: int = 128):
        self.vocab = vocab
        self.seq_len = seq_len
        self.corpus_path = corpus_path
        self.lines = pd.read_csv(corpus_path, sep='\t', names=['text_a', 'text_b', 'label'])
        self.corpus_lines = self.lines.shape[0]

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        t1, t2, is_next_label = self.get_sentence(idx)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        t1 = [self.vocab['[CLS]']] + t1_random + [self.vocab['[SEP]']]
        t2 = t2_random + [self.vocab['[SEP]']]
        t1_label = [self.vocab['[PAD]']] + t1_label + [self.vocab['[PAD]']]
        t2_label = t2_label + [self.vocab['[PAD]']]

        segment_label = ([0 for _ in range(len(t1))] + [1 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        attention_mask = len(bert_input) * [1] + len(padding) * [0]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)
        attention_mask = np.array(attention_mask)
        bert_input = np.array(bert_input)
        segment_label = np.array(segment_label)
        bert_label = np.array(bert_label)
        is_next_label = np.array(is_next_label)
        output = {"input_ids": bert_input,
                  "token_type_ids": segment_label,
                  'attention_mask': attention_mask,
                  "bert_label": bert_label}, is_next_label
        return output

    def random_word(self, sentence):
        import random
        tokens = sentence.split()
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80%
                if prob < 0.8:
                    tokens[i] = self.vocab['[MASK]']

                # 10%
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10%
                else:
                    tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])

                output_label.append(self.vocab.get(token, self.vocab['[UNK]']))

            else:
                tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                output_label.append(-100)
        return tokens, output_label

    def get_sentence(self, idx):

        t1, t2, _ = self.lines.iloc[idx].values
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.lines.iloc[random.randrange(self.lines.shape[0])].values[1], 0


vocab = pd.read_csv('Bert-vocab.txt', names=['word'])
vocab_dict = {}
for key, value in vocab.word.to_dict().items():
    vocab_dict[value] = key
pretrain_dataset = BERTDataset('./my_test/bert_pretrain_data/gaiic_track3_round1_train_20210228.tsv', vocab_dict, 64)
prevalid_dataset = BERTDataset('./my_test/bert_pretrain_data/gaiic_track3_round1_testA_20210228.tsv', vocab_dict, 64)
train_loader = DataLoader(pretrain_dataset, batch_size=64)
valid_loader = DataLoader(prevalid_dataset, batch_size=64)


def evaluate(model, data_loader, device='cuda'):
    model.eval()
    losses = []
    losses = []
    pbar = tqdm(data_loader)
    for data_label in pbar:
        data = data_label[0]
        next_sentence_label = data_label[1].to(device).long()

        input_ids = data['input_ids'].to(device).long()
        token_type_ids = data['token_type_ids'].to(device).long()
        attention_mask = data['attention_mask'].to(device).long()
        labels = data['bert_label'].to(device).long()
        optim.zero_grad()
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                        labels=labels, next_sentence_label=next_sentence_label)
        loss = outputs['loss']
        losses.append(loss.cpu().detach().numpy())
    loss = np.mean(losses)
    return loss


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = BertConfig(vocab_size=len(WORDS) + 1)
model = BertForPreTraining.from_pretrained('bert-base-chinese')
model = model.to(device)
# model=nn.DataParallel(model,device_ids=[0,1])
optim = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()
NUM_EPOCHS = 5
for epoch in range(NUM_EPOCHS):
    pbar = tqdm(train_loader)
    losses = []
    for data_label in pbar:
        data = data_label[0]
        next_sentence_label = data_label[1].to(device).long()

        input_ids = data['input_ids'].to(device).long()
        token_type_ids = data['token_type_ids'].to(device).long()
        attention_mask = data['attention_mask'].to(device).long()
        labels = data['bert_label'].to(device).long()
        optim.zero_grad()
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                        masked_lm_labels=labels, next_sentence_label=next_sentence_label)
        loss = outputs[0]
        losses.append(loss.cpu().detach().numpy())
        loss.backward()
        optim.step()
        pbar.set_description(f'epoch:{epoch} loss:{np.mean(losses)}')
    loss = evaluate(model, valid_loader)
    print('=*' * 50)
    print('valid loss:', loss)
    print('=*' * 50)
