import numpy as np
import re
import torch
from torch.utils.data import TensorDataset, random_split
import sys
import warnings
import os
import yaml
import metadata
warnings.filterwarnings('ignore')
#import metadata
from metadata.modules import read_jason_file as rf
from metadata.modules import preprocess as pr
from metadata.modules.load_data import Read
from metadata.models import custom_model as cm
from metadata.modules import config
from metadata.modules import *

working_folder = os.path.dirname(os.path.realpath(__file__))
try:
    meta_folder = yaml.safe_load(open(os.path.join(working_folder, 'config.yaml')))
    sample_data_path = meta_folder['sample_data_path']
    sample_data_path = os.path.join(working_folder, sample_data_path)
except Exception as exc:
    print('Attempted to read config file: {}',format(exc))
    sys.exit(1)
print(sample_data_path)

class news_qa_read():
    def __init__(self, text="Read_data"):
        self.doc_text = text
    
    def read_jason(file_path):
        qa_data=rf.load_json(file_path_=config['file_path'])
        qa_data_table  = rf.get_structure_data(data=qa_data)
        qa_data_table['Question1'] = qa_data_table['Question'].apply(rf.preprocess)
        rf.write_csv(qa_data_table, config['write_path'])
        return qa_data_table

    def load_data(path):
        path = sample_data_path
        questions,paragraph,st_index,en_index = Read.get_data(path=path)
        return questions,paragraph,st_index,en_index

    def preprocess_data(qa_data_table):
        #preprocess = pr.preprocess()
        #qa_data_table['Question1'] = qa_data_table.Question.apply(pr.preprocess)
        return qa_data_table
    
class news_qa_preprocess():
    def __init__(self, text="Read_data"):
        self.doc_text = text

    def preprocess_data(questions):
        pr.process_text()
        questions = questions.apply(pr.preprocess)
        return questions


if __name__ == '__main__':

    news_qa = news_qa_read()
    print("It's working!")
    questions,paragraph,st_index,en_index = news_qa.load_data()
    #print(questions,paragraph,st_index,en_index)
    print('Data is loaded')
    news_qa_preprocess = news_qa_preprocess(questions)
    print('preprocessed')
    cm.main(q_data=questions,p_data=paragraph,ans_st_index=st_index)
    print('modeled')




    

