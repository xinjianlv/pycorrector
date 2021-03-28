
import os
import logging
from flask import Flask, request

# 定义版本号（记得每次发布版本是更新）
version = "v1.0.0"

# 获取运行模式
run_mode = "local" if os.getenv('CONSUL_LOCAL') == '1' else "online"
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]-[%(threadName)s]-[%(filename)s:%(funcName)s:%(lineno)s]-%(levelname)s:  %(message)s' )
logger = logging.getLogger()


# 初始化日志模块
logger.info("################################## Server Start ##################################")
logger.info("################################# Version:%s #################################" % version)
logger.info("################################## RunMode:%s ##################################" % run_mode)

# 初始化配置管理器

# flask 相关
app = Flask(__name__)
pid = os.getpid()


@app.route('/test', methods=['POST'])
def test():
    logger.info("-_-_-_-_-_-_-_-_-_-_-_-_-_-_ %s %s" % (request, request.data))

    if request.form:  # ImmutableMultiDict([]) 也是 False
        request_data = request.form.to_dict()
    elif request.json:  # None 是 False
        request_data = request.get_json()
    else:  # 如果不在，一般在 request.data 中
        return {"status": 0, "remark": "Unknow Post Param"}

    logger.info("-_-_-_-_-_-_-_-_-_-_-_-_-_-_ %s" % request_data)

    return {'Test': "Hello Python Flask"}


# test = app.route('/test', methods=['POST'])(test)


if __name__ == "__main__":
    # 确定服务运行端口
    server_port = 8899
    is_debug = False
    if run_mode == 'local':
        is_debug = True
        server_port = 5000

    app.run(host='0.0.0.0', port=server_port, debug=is_debug)
