# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Run BERT on Masked LM.
"""


import argparse
import sys

sys.path.append('../..')
from transformers import pipeline


MASK_TOKEN = "[MASK]"

base_dir = './extract_bert/'
tokenizer_dir = '/Users/nocml/.pycorrector/datasets/bert_models/chinese_finetuned_lm'
# base_dir = './my_test/model_files/bert/'
# base_dir = '/Users/nocml/.pycorrector/datasets/bert_models/chinese_finetuned_lm'
def main():
    # parser = argparse.ArgumentParser()
    #
    # # Required parameters
    # parser.add_argument("--bert_model_dir", default=base_dir,
    #                     type=str,
    #                     help="Bert pre-trained model dir")
    # args = parser.parse_args()

    nlp = pipeline('fill-mask',
                   model=tokenizer_dir,
                   tokenizer=tokenizer_dir
                   # device=0  # gpu device id
                   )

    # i = nlp('hi lili, What is the name of the [MASK] ?')
    # print(i)
    #
    # i = nlp('今天[MASK]情很好')
    # print(i)
    #
    # i = nlp('少先队员[MASK]该为老人让座')
    # print(i)
    #
    # i = nlp('[MASK]七学习是人工智能领遇最能体现智能的一个分知')
    # print(i)
    #
    # i = nlp('机[MASK]学习是人工智能领遇最能体现智能的一个分知')
    # print(i)

    i = nlp('[MASK]约下周三下午2点现场面试@张诗苇')
    print(i)
if __name__ == "__main__":
    main()
