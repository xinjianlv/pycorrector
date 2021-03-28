import logging
from torch import nn
from transformers import BertModel,BertConfig

logger = logging.getLogger()

class BertClassificationModel(nn.Module):
    def __init__(self, cls, model_file):
        super(BertClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_file)
        logger.info('load model from path:%s'%model_file)
        self.dense = nn.Linear(768, cls)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_hidden_state = bert_output[0][:, :, :]
        linear_output = self.dense(bert_hidden_state)
        # out_put_prob = self.softmax(linear_output) # 损失函数使用交叉熵，所以这里不必使用softmax

        return linear_output


if __name__ == '__main__':
    import os
    from transformers import BertTokenizer
    pwd_path = os.path.abspath(os.path.dirname(__file__))
    print(pwd_path)
    tokenizer = BertTokenizer.from_pretrained(os.path.join(pwd_path,'model_files/bert'))
    print('size:', int(tokenizer.vocab_size))
    model = BertClassificationModel(cls=tokenizer.vocab_size, model_file=os.path.join(pwd_path,'model_files/bert'))

    input = '这是个测试。'

    input_encode = tokenizer.encode_plus(input, max_length=512, pad_to_max_length=True,return_tensors='pt')
    # print(input_encode)
    output = model.forward(input_ids=input_encode['input_ids'],attention_mask=input_encode['attention_mask'])
    print('out:', output.shape)
    print('end...')
