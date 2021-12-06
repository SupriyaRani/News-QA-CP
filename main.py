import numpy as np
import re
import torch
from torch.utils.data import TensorDataset, random_split
import sys
import warnings
import os
warnings.filterwarnings('ignore')
#import metadata
from metadata.modules import read_jason_file as rf
from metadata.modules import preprocess as pr
from metadata.modules.load_data import Read
from metadata.models import custom_model
from metadata.modules import config



class news_qa_read:
    def __init__(self, text="Read_data"):
        self.doc_text = text
    
    def read_jason(file_path):
        qa_data=rf.load_json(file_path_=config['file_path'])
        qa_data_table  = rf.get_structure_data(data=qa_data)
        qa_data_table['Question1'] = qa_data_table['Question'].apply(rf.preprocess)
        rf.write_csv(qa_data_table, config['write_path'])
        return qa_data_table

    def load_data(path):
        questions,paragraph,st_index,en_index = Read.get_data(path='data/sample_data.csv')
        return questions,paragraph,st_index,en_index

    def preprocess_data(qa_data_table):
        #preprocess = pr.preprocess()
        #qa_data_table['Question1'] = qa_data_table.Question.apply(pr.preprocess)
        return qa_data_table
    
class news_qa_preprocess(object):
    def __init__(self, text="Read_data"):
        self.doc_text = text

    def preprocess_data(questions):
        #preprocess = pr.preprocess()
        questions = questions.apply(pr.preprocess)
        return questions


if __name__ == '__main__':

    news_qa = news_qa_read()
    print("It's working!")
    questions,paragraph,st_index,en_index = news_qa.load_data()
    print(questions,paragraph,st_index,en_index)
    news_qa_preprocess = news_qa_preprocess(questions)
    print(news_qa_preprocess)
    

