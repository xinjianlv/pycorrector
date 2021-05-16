#coding=utf-8
import pdb
import os,sys
import docx
import re

def run():
    contents = []
    for root, dirs, files in os.walk('./data/童话'):
        for f in files:
            if f.endswith('docx'):
                document = docx.Document(os.path.join(root,f))
                for para in document.paragraphs:
                    lines = para.text.split('\n')
                    for l in lines:
                        if len(l.strip()) > 0:
                            contents.append(l.strip() + '\n')
                contents.append('__file__\n')
            elif f.endswith('txt'):
                document = open(os.path.join(root,f),'r', encoding='utf-8')
                doc_list = document.read().split('\n\n\n')
                for doc in doc_list:
                    lines = doc.split('\n')
                    for l in lines:
                        if len(l.strip()) > 0:
                            contents.append(l.strip() + '\n')

                    contents.append('__file__\n')
                    # if len(l.strip()) > 0:
                    #     contents.append(l.strip() + '\n')


    fw = open('./data/student/part2.txt','w')
    for line in contents:
        line = line.strip()
        if '__file__' in line:
            fw.write('\n')
        else:
            if len(line.strip('\n')) > 0:
                fw.write(line + '\n')
    #
    fw.flush()
    fw.close()


if __name__ == '__main__':
    run()