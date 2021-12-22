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

import spacy
from tqdm import tqdm
import re
import time
import pickle
pd.set_option('display.max_colwidth', 200)
import tensorflow_hub as hub
import tensorflow as tf

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

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

    #elmo
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

    # just a random sentence
    x = ["Roasted ants are a popular snack in Columbia"]

    # Extract ELMo features 
    embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

    embeddings.shape

    def elmo_vectors(x):
        embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            # return average of ELMo features
            return sess.run(tf.reduce_mean(embeddings,1))
    train = paragraph
    list_train = [train[i:i+100] for i in range(0,train.shape[0],100)]
    #list_test = [test[i:i+100] for i in range(0,test.shape[0],100)]

    # Extract ELMo embeddings
    elmo_train = [elmo_vectors(x['clean_tweet']) for x in list_train]
    #elmo_test = [elmo_vectors(x['clean_tweet']) for x in list_test]

    elmo_train_new = np.concatenate(elmo_train, axis = 0)
    #elmo_test_new = np.concatenate(elmo_test, axis = 0)

    #save elmo_train_new
    pickle_out = open("elmo_train_03032019.pickle","wb")
    pickle.dump(elmo_train_new, pickle_out)
    pickle_out.close()

    # save elmo_test_new
    #pickle_out = open("elmo_test_03032019.pickle","wb")
    #pickle.dump(elmo_test_new, pickle_out)
    #pickle_out.close()

    # load elmo_train_new
    pickle_in = open("elmo_train_03032019.pickle", "rb")
    elmo_train_new = pickle.load(pickle_in)

    # load elmo_train_new
    #pickle_in = open("elmo_test_03032019.pickle", "rb")
    #elmo_test_new = pickle.load(pickle_in)

    from sklearn.model_selection import train_test_split

    xtrain, xvalid, ytrain, yvalid = train_test_split(elmo_train_new, 
                                                    train['label'],  
                                                    random_state=42, 
                                                    test_size=0.2)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    lreg = LogisticRegression()
    lreg.fit(xtrain, ytrain)

    preds_valid = lreg.predict(xvalid)

    f1score = f1_score(yvalid, preds_valid)
    print(f1score)
    # make predictions on test set
    #preds_test = lreg.predict(elmo_test_new)
    
    