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

##lib
import pandas as pd
import numpy as np
#import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
#from keras.models import Sequential
#from keras.layers.recurrent import LSTM, GRU
#from keras.layers.core import Dense, Activation, Dropout
#from keras.layers.embeddings import Embedding
#from keras.layers.normalization import BatchNormalization
#from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
#from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
#from keras.preprocessing import sequence, text
#from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, Embedding
from tensorflow.keras.layers import SpatialDropout1D, Dense, Dropout, Input, concatenate, Conv1D, Activation, Flatten
from sklearn.preprocessing import LabelEncoder

FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
LOWER_CASE = True
MAX_LEN = 300
EMBED_SIZE = 200
NUM_WORDS=20000

#####

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
    print("Main initiated!")
    questions,paragraph,st_index,en_index = news_qa.load_data()
    print(questions,paragraph,st_index,en_index)
    print('Data is loaded')
    news_qa_preprocess = news_qa_preprocess(questions)
    print('preprocessed')
    cm.main(q_data=questions,p_data=paragraph,ans_st_index=st_index)
    print('modeled')
    
    