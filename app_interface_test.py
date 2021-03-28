#导入模块
import web
import pdb
from web import form
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]-[%(threadName)s]-[%(filename)s:%(funcName)s:%(lineno)s]-%(levelname)s:  %(message)s' )
logger = logging.getLogger()
#模板
render = web.template.render('templates/')

#url映射
urls = ('/', 'index')

#表单
InputForm = form.Form(
    # form.Textbox(input_id="foo",name="input_text",value='这是一个测试。'),
    form.Textarea(name='input_sentences',value='少先队员因该为老人让坐\r\n少 先  队 员 因 该 为 老人让坐\r\n机七学习是人工智能领遇最能体现智能的一个分知\r\n到以深切的问候',style="margin: 0px; width: 310px; height: 187px;"),
    # form.Textbox("bax",
    #     form.notnull,
    #     form.regexp('\d+', 'Must be a digit'),
    #     form.Validator('Must be more than 5', lambda x:int(x)>5)),
    form.Textarea(name='output_sentences',style="margin: 0px; width: 310px; height: 187px;"),
    # form.Checkbox('curly'),
    # form.Dropdown('french', ['mustard', 'fries', 'wine'])
    )

RetFrom = form.Form(
    form.Textarea(name='output_sentences')
)

from pycorrector.bert.bert_corrector import BertCorrector
d = BertCorrector()

class index:
    def GET(self):
        form = InputForm()
        # 确保通过调用它来创建表单的副本（上面一行）
        # 否则更改将全局显示
        # return form.read()
        return render.formtest(form)

    def POST(self):
        inputform = InputForm()
        if not inputform.validates():
            return render.formtest(inputform)
        else:
            # form.d.boe和form ['boe'].value是等价的方式
            # 从表单中提取经过验证的参数。
            sents = inputform.d.input_sentences.split('\r\n')

            r_cents = []
            for sent in sents:
                corrected_sent, err = d.bert_correct(sent)
                r_cents.append(corrected_sent)

            rtform = form.Form(
                # form.Textbox(input_id="foo", name="input_text", value='这是一个测试。'),
                form.Textarea(name='input_sentences',value=inputform.d.input_sentences,style="margin: 0px; width: 310px; height: 187px;"),
                form.Textarea(name='output_sentences',value='\r\n'.join(r_cents),style="margin: 0px; width: 310px; height: 187px;")
                )
            return render.formtest(rtform()) # open(r'./templates/1.html').read() #"Grrreat success! boe: %s, bax: %s" % (form.d.input_text, form.get('input_text').value)


if __name__=="__main__":
    web.internalerror = web.debugerror
    app = web.application(urls, globals())
    app.run()