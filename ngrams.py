import numpy as np
import pandas as pd
from typing import List, Tuple
from nltk import word_tokenize
from collections import Counter
from tqdm import tqdm

Corpus = List[str]


def corpus_from_file(file_path: str) -> Corpus:
    with open(file_path, "r") as file:
        return file.readlines()


class Ngram:
    def __init__(self, n: int):
        self.n = n
        # Matriz de MxP donde M=numero de (N-1)-gramas del corpus
        # y P=size del vocabulario
        self.frequency_table = None
        self.vocav = set()

    def fit(self, corpus: Corpus):
        all_n_grams = []

        for text in tqdm(corpus):
            tokens = self.tokenize(text)
            n_grams = self.extract_ngrams(tokens)
            all_n_grams.append(n_grams)

        # reemplazar por np.array
        self.frequency_table = None
        self.create_frequency_table(all_n_grams)

    def extract_ngrams(self, tokens):
        # Extract all the self.n-grams of the text.
        n_grams = []
        for i in range(len(tokens) - self.n + 1):
            n_grams.append(tuple(tokens[i:i + self.n]))

        return n_grams

    def create_frequency_table(self, n_grams):
        # Create a frequency table from the n-grams

        print("Creating frequency table")
        print("Flattening n-grams")
        n_grams = [item for sublist in n_grams for item in sublist]
        print("Creating vocabulary")
        self.vocav = set(n_grams)
        print(f"Vocabulary size: {len(self.vocav)}")
        n_gram_counts = Counter(n_grams)
        print(f"Number of n-grams: {len(n_gram_counts)}")

        frequency_table = {}

        print("Creating frequency table")

        for i, n_gram in tqdm(enumerate(n_grams)):
            if n_gram[:-1] not in frequency_table:
                frequency_table[n_gram[:-1]] = {}
            if n_gram[-1] not in frequency_table[n_gram[:-1]]:
                frequency_table[n_gram[:-1]][n_gram[-1]] = 0

            frequency_table[n_gram[:-1]][n_gram[-1]] += 1

        self.frequency_table = frequency_table

    def tokenize(self, text):
        text = text.lower()
        text = text.split(" ")
        text = ["<s>"] * (self.n - 1) + text + ["</s>"]
        return text

    def predict(self, context: Tuple[str]) -> str:
        # return the token with highest probability given n-1 tokens

        tokenized_context = self.tokenize(context)[:-1]
        print(tokenized_context)
        n_minus_1_gram = tuple(tokenized_context[-(self.n - 1):])

        print(n_minus_1_gram)

        if n_minus_1_gram not in self.frequency_table:
            return None

        next_token = max(self.frequency_table[n_minus_1_gram], key=self.frequency_table[n_minus_1_gram].get)
        return next_token

    def perplexity(self, text: Corpus) -> float:
        # Calculate the perplexity of the text given the n-gram model.
        exponent = 0

        for sentence in text:
            tokens = self.tokenize(sentence)
            n_grams = self.extract_ngrams(tokens)

            for n_gram in n_grams:
                context = n_gram[:-1]
                token = n_gram[-1]

                if context not in self.frequency_table:
                    continue

                if token not in self.frequency_table[context]:
                    continue

                exponent += np.log2(self.frequency_table[context][token])

        perplexity = np.exp(-exponent / len(text))
        return perplexity

    def bleu_score(self, reference: Corpus, candidate: Corpus) -> float:
        # Calculate the BLEU score between the reference and candidate corpora.
        pass


if __name__ == "__main__":
    ngram = Ngram(3)
    corpus = corpus_from_file("corpus.txt")
    ngram.fit(corpus)
    print(ngram.frequency_table)
    print(ngram.vocav)