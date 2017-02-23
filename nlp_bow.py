import json
import gzip
from collections import Counter
from spacy.en import English
import scipy.sparse as sp
from scipy.io import mmwrite
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def transcript_stream(filename):
    ''' stream over lines in jsonl file '''
    for i, jsonl in enumerate(gzip.open(filename)):
        line = jsonl.decode('utf-8')
        yield (i, json.loads(line)['transcript'])


TEDSTOPS = set(['applause', 'cheers', 'laughter'])


def is_stop(token):
    ''' return true if is stop, or is in TEDSTOPS '''
    return any([token.is_stop, token.text in TEDSTOPS])


def select_token(token):
    ''' return false if token should be excluded '''
    return not any([token.is_space, token.is_punct,
                    is_stop(token), token.is_digit])


def word_tokens(docs, language_mdl):
    ''' return docs' vocabulary '''
    return (token.text
            for _, doc in docs
            for token in language_mdl(doc))


def build_vocabulary(docs, nlp):
    ''' build term frequency from docs corpus
    '''
    return Counter(word_tokens(docs, nlp))


def doc_term_matrix_row(docs, language_mdl):
    ''' yield doc id and token counts for each document '''
    for i, doc in docs:
        yield i, Counter(token.orth for token in language_mdl(doc)
                         if select_token(token))


def tfidf_weight(doc_term_matrix):
    ''' tf-idf weight document frequency matrix (smooth)
    '''
    n_docs, n_terms = doc_term_matrix.shape
    df = np.bincount(doc_term_matrix.indices, minlength=n_terms) + 1
    idf = np.log((n_docs + 1) / df) + 1.0
    return doc_term_matrix.dot(sp.diags(idf, 0))


def doc_term_matrix(docs, nlp):
    ''' build tfidf representation of docs corpus
    '''
    doc_term_mm = ((i, k, v)
                   for i, row in doc_term_matrix_row(docs, nlp)
                   for k, v in row.items())
    rows, cols, vals = zip(*doc_term_mm)
    return sp.coo_matrix((vals, (rows, cols)), dtype=int).tocsr()


if __name__ == '__main__':

    TALKPATH = 'data/tedtalk.jl.gz'
    nlp = English()

    # vocab = build_vocabulary(transcript_stream(TALKPATH), nlp)
    # print(vocab.most_common(10))

    tfidf = tfidf_weight(doc_term_matrix(transcript_stream(TALKPATH), nlp))
    d = cosine_similarity(tfidf, tfidf)
    mmwrite('data/tedsim.mtx', d)
    print(d.shape)
