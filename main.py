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


class news_qa:
    def __init__(self, text="Hi"):
        self.doc_text = text
    
    def read_jason(file_path):
        file_path='data/combined-newsqa-data-v1.json'
        write_path = 'data/qa_data.csv'
        qa_data=rf.load_json(file_path_=file_path)
        qa_data_table  = rf.get_structure_data(data=qa_data)
        qa_data_table['Question1'] = qa_data_table['Question'].apply(rf.preprocess)
        rf.write_csv(qa_data_table, write_path)
        return qa_data_table
        
    def preprocess_data(qa_data_table):
        #preprocess = pr.preprocess()
        #qa_data_table['Question1'] = qa_data_table.Question.apply(pr.preprocess)
        return qa_data_table
    

if __name__ == '__main__':

    news_qa = news_qa()
    print("It's working!")
    qa_data= news_qa.read_jason()
    print(qa_data['Question'].head())
    
