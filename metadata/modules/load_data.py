

import pandas as pd
import re


class Read():
    def __init__(self):
        pass

    @staticmethod
    def get_data(path):

        data = pd.read_csv(path)
        print(data.head())
        data = data.dropna()

        print(data.shape)
        questions =data.Question1.str.lower().tolist()
        paragraph = data.paragraph.str.lower().tolist()
        st_index = data.ans_st_index.tolist()
        en_index = data.ans_en_index.tolist()

        #questions_pr = [q.split("\n") for q in questions]
        #questions_pr = [' ' + str(pref) + '.' for pref in questions_pr]

        #paragraph_pr = paragraph.split("\n")
        #paragraph_pr = [' ' + str(pref) + '.' for pref in paragraph_pr]


        """We will need each word and other symbol that we want to keep to be in lower case and separated by spaces so we can "tokenize" them."""

        """## Cleaning data
        Getting the non_breaking_prefixes as a clean list of words with a point at the end so it is easier to use.
        """
        return(questions,paragraph,st_index,en_index)



