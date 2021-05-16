

import os
import pdb
import json

def load_files(file):

    contents = []
    for root, dirs, files in os.walk(file):
        for file in files:
            print(file)

            if file.endswith('self.json'):
                contents.append(file)
                with open(os.path.join(root,file),'r') as fr:
                    jdata = json.loads(fr.readline())
                    ct = ''
                    for sub in jdata['data']:
                        cx = sub['cx']
                        cy = sub['cy']
                        h = sub['h']
                        ct = sub['text']
                        print(cx,cy,h)
                        contents.append(ct)
    return contents


if __name__ == '__main__':
    error_sentences = load_files('./my_test/data/OK_rec_raw/')
    fw = open('./my_test/data/ocr_contents.txt','w')
    for line in error_sentences:
        arr = line.split('。')
        for sub in arr:
            fw.write(sub + '\n')
    fw.flush()
    fw.close()

