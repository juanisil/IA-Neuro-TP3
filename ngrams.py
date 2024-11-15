import numpy as np
import pandas as pd
from typing import List, Tuple
from nltk import word_tokenize
from collections import Counter
from tqdm import tqdm

Corpus = List[str]
START = "<s>"
END = "</s>"


def corpus_from_file(file_path: str) -> Corpus:
    with open(file_path, "r") as file:
        return file.readlines()


class Ngram:
    def __init__(self, n: int):
        self.n = n
        # dict(dict())
        self.frequency_table = None
        self.vocav = set()

    def fit(self, corpus: Corpus, verbose=False):
        all_n_grams = []

        iterator = corpus
        if verbose:
            iterator = tqdm(iterator, total=len(corpus))
            print("Tokenizing corpus")
        for text in iterator:
            tokens = self.tokenize(text)
            n_grams = self.extract_ngrams(tokens, verbose)
            all_n_grams.append(n_grams)

        # reemplazar por np.array
        self.frequency_table = None
        self.create_frequency_table(all_n_grams, verbose)

    def extract_ngrams(self, tokens, verbose=False):
        # Extract all the self.n-grams of the text.
        n_grams = []
        iterator = range(len(tokens) - self.n + 1)
        if verbose:
            iterator = tqdm(iterator, total=len(tokens) - self.n + 1)
            print("Extracting n-grams")
        for i in iterator:
            n_grams.append(tuple(tokens[i:i + self.n]))

        return n_grams

    def create_frequency_table(self, n_grams, verbose=False):
        # Create a frequency table from the n-grams

        if verbose:
            print("Creating frequency table")
            print("Flattening n-grams")
        n_grams = [item for sublist in n_grams for item in sublist]
        if verbose:
            print("Creating vocabulary")
        self.vocav = set(n_grams)
        if verbose:
            print(f"Vocabulary size: {len(self.vocav)}")
        n_gram_counts = Counter(n_grams)
        if verbose:
            print(f"Number of n-grams: {len(n_gram_counts)}")

        frequency_table = {}

        if verbose:
            print("Creating frequency table")

        freq_t_iterator = enumerate(n_grams)
        if verbose:
            freq_t_iterator = tqdm(freq_t_iterator, total=len(n_grams))
        for i, n_gram in freq_t_iterator:
            if n_gram[:-1] not in frequency_table:
                frequency_table[n_gram[:-1]] = {}
            if n_gram[-1] not in frequency_table[n_gram[:-1]]:
                frequency_table[n_gram[:-1]][n_gram[-1]] = 0

            frequency_table[n_gram[:-1]][n_gram[-1]] += 1

        self.frequency_table = frequency_table

    def tokenize(self, text):
        text = text.lower()
        text = text.split(" ")
        text = [START] * (self.n - 1) + text + [END]
        return text

    def predict(self, context: str) -> str:
        # return the token with highest probability given n-1 tokens

        tokenized_context = self.tokenize(context)[:-1]
        n_minus_1_gram = tuple(tokenized_context[-(self.n - 1):])
        if n_minus_1_gram not in self.frequency_table:
            return None

        next_token = max(self.frequency_table[n_minus_1_gram], key=self.frequency_table[n_minus_1_gram].get)
        return next_token

    def generate(self, context: Tuple[str]="") -> str:
        max_length = 30
        answer = ""
        next_word = ""
        while(next_word != END and max_length>0):
            next_word = self.predict(context)
            answer += f" {next_word}"
            context = context[1:] + next_word
            max_length -= 1

        return answer[1:-len(END)]


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
