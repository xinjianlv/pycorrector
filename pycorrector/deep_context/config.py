# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: network configuration
"""
import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

# Training data path.
# chinese corpus
raw_train_paths = [
    os.path.join(pwd_path, '../data/cn/CGED/CGED18_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED17_HSK_TrainingSet.xml'),
    os.path.join(pwd_path, '../data/cn/CGED/CGED16_HSK_TrainingSet.xml'),
    # os.path.join(pwd_path, '../data/cn/CGED/sample_HSK_TrainingSet.xml'),
]

output_dir = os.path.join(pwd_path, 'output')

train_path = os.path.join(output_dir, 'train.txt')
# Validation data path.
test_path = os.path.join(output_dir, 'test.txt')

model_dir = os.path.join(output_dir, 'model_files')
model_path = os.path.join(model_dir, 'model.pth')
vocab_path = os.path.join(model_dir, 'vocab.txt')

# nets
word_embed_size = 200
hidden_size = 200
n_layers = 2
use_mlp = True
dropout = 0.0

# train
maxlen = 400
epochs = 20
batch_size = 64
min_freq = 1
ns_power = 0.75
learning_rate = 1e-3
gpu_id = 0

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
