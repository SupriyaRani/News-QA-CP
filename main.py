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



class news_qa:
    def __init__(self, text="Hi"):
        self.doc_text = text
    
    def read_jason(file_path):
        qa_data=rf.load_json(file_path_=config['file_path'])
        qa_data_table  = rf.get_structure_data(data=qa_data)
        qa_data_table['Question1'] = qa_data_table['Question'].apply(rf.preprocess)
        rf.write_csv(qa_data_table, config['write_path'])
        return qa_data_table

    def load_data(path):
        questions,paragraph,st_index,en_index = Read.get_data(path=config['sample_data_path'])
        return questions,paragraph,st_index,en_index

    def preprocess_data(qa_data_table):
        #preprocess = pr.preprocess()
        #qa_data_table['Question1'] = qa_data_table.Question.apply(pr.preprocess)
        return qa_data_table
    

if __name__ == '__main__':

    news_qa = news_qa()
    print("It's working!")
    #questions,paragraph,st_index,en_index= news_qa.load_data()
    questions,paragraph,st_index,en_index = news_qa.load_data()
    print(questions,paragraph,st_index,en_index)
    
