import json
import pandas as pd
import numpy as np
import os

def load_json(file_path_):
    data = {}
    #file_path_='data/combined-newsqa-data-v1.json'
    if os.path.exists(file_path_):
        with open(file_path_, 'r', encoding="cp866") as input_file:
            file = input_file.read()
            data = json.loads(file)
    else:
        raise FileExistsError('It seems given path does not have required question answer file.')
    return(data)

def get_structure_data(data):
    final_data = pd.DataFrame()
    if len(data)>0:
        paragraph = []
        questions=[]
        ans_st_index=[]
        ans_en_index=[]
        story_id = []
        for item in data['data']:
            for item_ in item['questions']:
                paragraph.append(item['text'])
                story_id.append(item['storyId'])
                questions.append(item_['q'])
                ans_st_index.append(item_['consensus'].get('s'))
                ans_en_index.append(item_['consensus'].get('e'))
        final_data = pd.DataFrame({'story_id':story_id,'Question':questions,'paragraph':paragraph
                      ,'ans_st_index':ans_st_index,'ans_en_index':ans_en_index})
    return(final_data)

from gensim.parsing.preprocessing import preprocess_string,strip_punctuation, remove_stopwords, strip_multiple_whitespaces
import re

def preprocess(sent):
    get_char = lambda x : re.sub('[^a-zA-Z0-9]',' ',x)
    CUSTOM_FILTERS = [lambda x:x.lower(), get_char, strip_multiple_whitespaces]
    output = preprocess_string(sent, CUSTOM_FILTERS)
    return(' '.join(output))

def write_csv(qa_data_table,file):
    qa_data_table.to_csv(file,index=False)

