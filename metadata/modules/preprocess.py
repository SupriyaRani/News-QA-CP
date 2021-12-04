from gensim.parsing.preprocessing import preprocess_string,strip_punctuation, remove_stopwords, strip_multiple_whitespaces
import re

def preprocess(sent):
    get_char = lambda x : re.sub('[^a-zA-Z0-9]',' ',x)
    CUSTOM_FILTERS = [lambda x:x.lower(), get_char, strip_multiple_whitespaces]
    output = preprocess_string(sent, CUSTOM_FILTERS)
    return(' '.join(output))