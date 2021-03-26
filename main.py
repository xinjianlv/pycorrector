
import logging
from my_test.main import train



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s]-[%(threadName)s]-[%(filename)s:%(funcName)s:%(lineno)s]-%(levelname)s:  %(message)s'
                        )
    train()