# _*_ coding: utf-8 _*_
# @Time : 19.2.21 20:51
# @Author : XJ.LYU(xinjian.lv@gmail.com)
# @Version：V 0.1
# @File : extract_new.py
# @desc :

import logging

import os
import pdb
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader,TensorDataset


logger = logging.getLogger()



class BertTool(object):


    @staticmethod
    def get_single_loader_from_raw_text(src_lines, trg_lines, tokenizer:BertTokenizer, batch_size):

        tensor_datasets = {'train': [], 'valid': []}
        input_ids = []
        token_type_ids = []
        attention_mask = []
        label_list = []

        assert len(src_lines) == len(trg_lines)

        for ndx, row in enumerate(src_lines):
            if ndx % 100 == 0:
                logger.info('%d/%d processed.' % (ndx, len(src_lines)))

            src = src_lines[ndx]
            trg = trg_lines[ndx]

            output = tokenizer.encode_plus(src, max_length=512, pad_to_max_length=True)
            input_ids.append(output['input_ids'])
            # token_type_ids.append(output['token_type_ids'])
            attention_mask.append(output['attention_mask'])

            labels_encode = tokenizer.encode_plus(trg, max_length=512, pad_to_max_length=True)
            label_list.append(labels_encode['input_ids'])

        data_set = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label_list}
        train_max_num = int(len(data_set['input_ids']))

        for name in ['input_ids', 'attention_mask', 'labels']:
            tensor_datasets['train'].append(torch.LongTensor(data_set[name][0:train_max_num]))

        train_data_set = TensorDataset(*tensor_datasets['train'])
        train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)

        return train_data_loader

    @staticmethod
    def get_loaders(dataset_path, tokenizer_model, batch_size):
        train_loader = BertTool.get_raw_text_loader(os.path.join(dataset_path,'train.src'),
                                                    os.path.join(dataset_path,'train.trg'),
                                                    tokenizer_model,
                                                    batch_size)
        valid_loader = BertTool.get_raw_text_loader(os.path.join(dataset_path, 'valid.src'),
                                                    os.path.join(dataset_path, 'valid.trg'),
                                                    tokenizer_model,
                                                    batch_size)
        return train_loader, valid_loader

    @staticmethod
    def get_raw_text_loader(src_file,trg_file,tk_model,batch_size):
        fr_src = open(src_file)
        fr_trg = open(trg_file)
        src_lines = fr_src.readlines()
        trg_lines = fr_trg.readlines()
        fr_src.close()
        fr_trg.close()

        return BertTool.get_single_loader_from_raw_text(src_lines, trg_lines, tk_model,batch_size)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s]-[%(threadName)s]-[%(filename)s:%(funcName)s:%(lineno)s]-%(levelname)s:  %(message)s'
                        )
    pwd_path = os.path.abspath(os.path.dirname(__file__))

    fr_src = open('./my_test/data/nlpcc2018+hsk/valid.src','r')
    fr_trg = open('./my_test/data/nlpcc2018+hsk/valid.trg','r')

    src_lines = fr_src.readlines()
    trg_lines = fr_trg.readlines()

    valid_loader = BertTool.get_single_loader_from_raw_text(src_lines, trg_lines, './my_test/model_files/bert/')
