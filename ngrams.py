import numpy as np
import pandas as pd
from typing import List, Tuple
from nltk import word_tokenize
from collections import Counter
from tqdm import tqdm

Corpus = List[str]
SENTENCE_START = "<l>"
SENTENCE_END = "</l>"

SCENE_START = "<s>"
SCENE_END = "</s>"

EPISODE_START = "<e>"
EPISODE_END = "</e>"

def corpus_from_file(file_path: str) -> Corpus:
    with open(file_path, "r") as file:
        return file.readlines()

class Ngram:
    def __init__(self, n: int):
        self.n = n
        # dict(dict())
        self.frequency_table = None
        self.vocav = set()

    def fit(self, corpus: Corpus, verbose=False, type="text"):
        """
        Types are:
            text: for normal text
            script: for script, with scene and episode markers
        """
        all_n_grams = []
        all_tokens = []

        iterator = corpus
        if verbose:
            iterator = tqdm(iterator, total=len(corpus))
            print("Tokenizing corpus")
        scene_started = False
        episode_started = False
        for text in iterator:
            if text == "\n":
                continue
            tokens = self.tokenize(text)
            if not episode_started:
                tokens = [EPISODE_START] + tokens
                episode_started = True
            if ("[scene" in text) or ("[Scene" in text):
                if scene_started:
                    tokens = tokens + [SCENE_END]
                tokens = [SCENE_START] + tokens
                scene_started = True
            if "\nEnd\n" in text:
                tokens = tokens + [SCENE_END, EPISODE_END]
                scene_started = False
                episode_started = False

            n_grams = self.extract_ngrams(tokens, verbose)
            all_n_grams.append(n_grams)
            all_tokens.append(tokens)

        self.frequency_table = None
        self.create_frequency_table(all_n_grams, verbose)

        return all_tokens

    def extract_ngrams(self, tokens, verbose=False):
        # Extract all the self.n-grams of the text.
        n_grams = []
        iterator = range(len(tokens) - self.n + 1)
        if verbose:
            iterator = tqdm(iterator, total=len(tokens) - self.n + 1)
            print("Extracting n-grams")
        for i in iterator:
            n_grams.append(tuple(tokens[i : i + self.n]))

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
        text = [SENTENCE_START] * (self.n - 1) + text + [SENTENCE_END]
        return text

    def predict(self, tokenized_context: str) -> str:
        # return the tokens distribution given the context
        # tokenized_context = self.tokenize(context)[:-1]
        n_minus_1_gram = tuple(tokenized_context[-(self.n - 1):])
        if n_minus_1_gram not in self.frequency_table:
            return None

        logits = self.frequency_table[n_minus_1_gram]
        return logits

    def generate(
        self, context: Tuple[str] = "", temperature: float = 0.9, max_length=30
    ) -> str:
        answer = ""
        if context == "":
            context = [SENTENCE_START] * (self.n - 1)
        else:
            context = self.tokenize(context)[:-1]
        next_word = ""
        while next_word != SENTENCE_END and max_length > 0:
            logits = self.predict(context)
            if logits is None:
                break

            logits = {k: v for k, v in logits.items() if v > 0}
            logits = {k: v ** (1 / temperature) for k, v in logits.items()}
            logits = {k: v / sum(logits.values()) for k, v in logits.items()}
            next_word = np.random.choice(list(logits.keys()), p=list(logits.values()))
            answer += f" {next_word}"
            context = context[1:] + [
                next_word,
            ]
            max_length -= 1

        return answer

        # next_word = ""
        # while(next_word != END and max_length>0):
        #     next_word = self.predict(context)
        #     answer += f" {next_word}"
        #     context = context[1:] + next_word
        #     max_length -= 1

        # return answer[1:-len(END)]

    def generate_scene(self, context: Tuple[str] = "", temperature: float = 0.9, max_sentences=10) -> str:
        scene = context
        while SCENE_END not in scene and max_sentences > 0: 
            sentence = self.generate(context, temperature)
            if SCENE_START in sentence:
                break
            scene += sentence
            context = sentence[-2:]

            max_sentences -= 1

        if SCENE_END not in scene:
            scene += SCENE_END
        return scene

    def generate_title(self, temperature: float = 0.9) -> str:
        title = self.generate("The One with the", temperature, 4)
        return EPISODE_START + "The One with the" + title

    def generate_episode(self, context: Tuple[str] = "", temperature: float = 0.9, max_scenes=10, max_lines_per_scene=10) -> str:
        episode = self.generate_title(temperature)
        context = "<s><l><l><l>[Scene: " + context
        i = 0

        while EPISODE_END not in episode and max_scenes > 0:
            if episode[-2:] == SCENE_END:
                context += "<s><l><l><l>"
            sentence = self.generate_scene(context, temperature, max_lines_per_scene)
            if i == 0:
                sentence = sentence.replace(SENTENCE_START, "\n" + SENTENCE_START)
            if EPISODE_START in sentence:
                break
            episode += sentence
            context = sentence[-2:]

            max_scenes -= 1
            i += 1

        return episode

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

    def precision(self, reference: Corpus, candidate: Corpus) -> float:
        # Calculate the precision of the candidate corpus given the reference corpus.
        correct = 0
        total = 0

        for ref, cand in zip(reference, candidate):
            ref_tokens = word_tokenize(ref)
            cand_tokens = word_tokenize(cand)

            for token in cand_tokens:
                if token in ref_tokens:
                    correct += 1

            total += len(cand_tokens)

        return correct / total

    def bleu_score(self, reference: Corpus, candidate: Corpus) -> float:
        # Calculate the BLEU score between the reference and candidate corpora.
        score = 0

        for ref, cand in zip(reference, candidate):
            ref_tokens = word_tokenize(ref)
            cand_tokens = word_tokenize(cand)

            for i in range(1, 5):
                ref_ngrams = self.extract_ngrams(ref_tokens, i)
                cand_ngrams = self.extract_ngrams(cand_tokens, i)

                ref_ngrams = Counter(ref_ngrams)
                cand_ngrams = Counter(cand_ngrams)

                correct = 0
                total = 0

                for ngram in cand_ngrams:
                    correct += min(cand_ngrams[ngram], ref_ngrams[ngram])

                total += len(cand_ngrams)

                score += correct / total

        score /= len(reference)


if __name__ == "__main__":
    ngram = Ngram(3)
    corpus = corpus_from_file("corpus.txt")
    ngram.fit(corpus)
    print(ngram.frequency_table)
    print(ngram.vocav)
