import numpy as np
import pandas as pd
from typing import List, Tuple
from nltk import word_tokenize
from collections import Counter

Corpus = List[str]
Token = str

class Ngram:
    def __init__(self, n:int):
        self.n = n
        self.frequency_table = pd.DataFrame()

    def fit(self, corpus:Corpus):
        all_n_grams = []
        for text in corpus:
            tokens: Token = word_tokenize(text.lower())
            n_grams = self.extract_ngrams(tokens)
            all_n_grams.append(n_grams)

        # reemplazar por np.array
        self.frequency_table = pd.DataFrame(Counter(all_n_grams).items(), columns=['N-gram', 'Frequency'])


    def extract_ngrams(self, text):
        # Extract all the self.n-grams of the text.
        pass
    
    def predict(self, prompt):
        # return the token with highest probability given n-1 tokens
        pass