import pandas as pd
import spacy
import numpy as np
from sklearn.datasets import load_files

from collections import OrderedDict
import pickle
import re

def insert_edge(M, v, w):
    """insert edge for undirected weighted graph
    """
    M[v][w] += 1
    M[w][v] += 1



def normalize(M):
    """normalize a matrix along column
    """
    sums = np.sum(M, axis=0)
    return np.divide(M, sums, where= sums != 0)

class Text_rank:
    """Text rank implementation, interface similar to scikit-learn.
    Attributes:
        WINDOW_SIZE: A window for calculating the connection between two words.
        m: the return value of get_word2pos_map, for storing the positions of occurances of each word.
        Gn: normalized transition matrix
        doc: spacy document object
        filter_list: a list contains target POS
        reg_obj: a regular expression object
    """

    def __init__(self, window_size=2, iteration=100, d=0.85, epsilon=1e-3, filtered_list=['NOUN'],
                 pattern=None, stopwords=None, verbose=False):
        self.WINDOW_SIZE = window_size
        self.iteration = iteration
        self.d = d
        self.epsilon = epsilon
        self.filtered_list = filtered_list
        self.verbose = verbose
    
    def build_edges(self, tokens, M, token2idx):
        """build edges for the graph.
        the edge is governed by the window size, if two words are within this width of the context, then
        insert an edge between the two word vertices

        :param tokens: list of tokens as strings
        :param window_size: the context to consider whether two words are connected,
        :return: Matrix M
        """
        window_size = self.WINDOW_SIZE

        for i in range(len(tokens) - window_size+1):
            span = tokens[i:i+window_size]
            center, contexts = span[0], span[1:]
            v = token2idx[center]
            for context in contexts:
                w = token2idx[context]
                insert_edge(M, v, w)

        
    def _text_rank(self, W):
        """the iteration step for text rank algorithm

        :param W: the transition matrix
        :param iteration: number of iteration
        :param d: coefficient for smoothing
        :param epsilon: criteria for stopping
        :return: the final state vector of shape (W.shape[1],1)
        """
        d = self.d
        epsilon = self.epsilon
        iteration = self.iteration
        s = np.ones((W.shape[1], 1)) #initial state set to all 1
        for i in range(iteration):
            s_new = (1 - d) + d * np.dot(W, s)
            # stop when absolute value differences between the old and new state are all below threshold
            if np.sum(np.abs(s - s_new)) < epsilon:
                if self.verbose:
                    print('finish in %dth iteration' %i)
                return s_new

            if self.verbose:
                print("iteration %d delta = %.3f" % (i, np.sum((s - s_new) ** 2)))
            s = s_new

        return s


    def fit(self, tokens):
        """The fitting step
        :return: the dict for the token weight pair
        """
        
        vocab = list(set(tokens))
        token2idx = {token: i for i, token in enumerate(vocab)}
        n = len(vocab)
        M = np.zeros((n, n))
        
        self.build_edges(tokens, M, token2idx)
        
        M = normalize(M)
        
        s = self._text_rank(M)
        s = s.ravel()
        return dict(sorted(zip(vocab, s), key = lambda x: x[1])[::-1])

if __name__ == '__main__':
    with open('./noun_verb_df.pickle','rb') as f:
        df = pickle.load(f)
    tokens = df.iloc[0]['nouns']
    tokens = tokens.split(' ')
    text_rank = Text_rank(pattern=None, window_size=4, stopwords=[])
    rank = text_rank.fit(tokens)
    print(rank)
    