from gensim.parsing.preprocessing import preprocess_string,strip_punctuation, remove_stopwords, strip_multiple_whitespaces
import re

import tensorflow as tf
import re
import joblib
import os
import json

def process_text(text, text_pr):
    tx_1 = text
    for prefix in text_pr:
        tx_1 = tx_1.replace(prefix, prefix + "###")
    tx_1 = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".###", tx_1)
    tx_1 = re.sub(r"\.###", '', tx_1)
    tx_1 = re.sub(r"  +", ' ', tx_1)
    tx_1 = tx_1.split("\n")
    return (tx_1)

def preprocess(sent):
    get_char = lambda x : re.sub('[^a-zA-Z0-9]',' ',x)
    CUSTOM_FILTERS = [lambda x:x.lower(), get_char, strip_multiple_whitespaces]
    output = preprocess_string(sent, CUSTOM_FILTERS)
    return(' '.join(output))

def load_json(file_path_):
    data = {}
    if os.path.exists(file_path_):
        with open(file_path_, 'r', encoding="cp866") as input_file:
            file = input_file.read()
            data = json.loads(file)
    else:
        raise FileExistsError('It seems given path does not have required question answer file.')
    return(data)

def load_pickle(file_name):
    token = joblib.load(filename=file_name)
    return(token)

def save_pikel(tokenizer,file_name):
    joblib.dump(value=tokenizer,filename=file_name)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
