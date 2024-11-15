import numpy as np
import pandas as pd
from typing import List, Tuple
from nltk import word_tokenize
from collections import Counter

Corpus = List[str]

class Ngram:
    def __init__(self, n:int):
        self.n = n
        # Matriz de MxP donde M=numero de (N-1)-gramas del corpus 
        # y P=size del vocabulario
        self.frequency_table = pd.DataFrame()
        self.vocav = set()

    def fit(self, corpus:Corpus):
        all_n_grams = []

        for text in corpus:
            tokens = self.tokenize(text)
            n_grams = self.extract_ngrams(tokens)
            all_n_grams.append(n_grams)

        # reemplazar por np.array
        self.frequency_table = None


    def extract_ngrams(self, tokens):
        # Extract all the self.n-grams of the text.
        pass

    def tokenize(self, text):
        text = text.lower()
        text = text.split(" ")
        text = ["<s>"] * (self.n-1) + text + ["</s>"]
        return text
    
    def predict(self, prompt):
        # return the token with highest probability given n-1 tokens
        pass