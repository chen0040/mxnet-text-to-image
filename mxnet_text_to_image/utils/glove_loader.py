import urllib.request
import os
import zipfile
import numpy as np
import logging
import pickle
import time

from mxnet_text_to_image.utils.download_utils import reporthook


def download_glove(data_dir_path, to_file_path):
    if not os.path.exists(to_file_path):
        if not os.path.exists(data_dir_path):
            os.makedirs(data_dir_path)

        glove_zip = data_dir_path + '/glove.6B.zip'

        if not os.path.exists(glove_zip):
            logging.debug('glove file does not exist, downloading from internet')
            urllib.request.urlretrieve(url='http://nlp.stanford.edu/data/glove.6B.zip', filename=glove_zip,
                                       reporthook=reporthook)

        logging.debug('unzipping glove file')
        zip_ref = zipfile.ZipFile(glove_zip, 'r')
        zip_ref.extractall(data_dir_path)
        zip_ref.close()


def load_glove(data_dir_path=None, embedding_dim=None):
    """
    Load the glove models (and download the glove model if they don't exist in the data_dir_path
    :param data_dir_path: the directory path on which the glove model files will be downloaded and store
    :param embedding_dim: the dimension of the word embedding, available dimensions are 50, 100, 200, 300, default is 100
    :return: the glove word embeddings
    """
    if embedding_dim is None:
        embedding_dim = 100

    glove_pickle_path = data_dir_path + "/glove.6B." + str(embedding_dim) + "d.pickle"
    if os.path.exists(glove_pickle_path):
        logging.info('loading glove embedding from %s', glove_pickle_path)
        start_time = time.time()
        with open(glove_pickle_path, 'rb') as handle:
            result = pickle.load(handle)
            duration = time.time() - start_time
            logging.debug('loading glove from pickle tooks %.1f seconds', (duration ))
            return result
    glove_file_path = data_dir_path + "/glove.6B." + str(embedding_dim) + "d.txt"
    download_glove(data_dir_path, glove_file_path)
    _word2em = {}
    logging.debug('loading glove embedding from %s', glove_file_path)
    file = open(glove_file_path, mode='rt', encoding='utf8')
    for i, line in enumerate(file):
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        _word2em[word] = embeds
        if i % 1000 == 0:
            logging.debug('loaded %d %d-dim glove words', i, embedding_dim)
    file.close()
    with open(glove_pickle_path, 'wb') as handle:
        logging.debug('saving glove embedding as %s', glove_pickle_path)
        pickle.dump(_word2em, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return _word2em


class GloveModel(object):
    """
    Class the provides the glove embedding and document encoding functions
    """
    model_name = 'glove-model'

    def __init__(self):
        self.word2em = None
        self.embedding_dim = None

    def load(self, data_dir_path, embedding_dim=None):
        if embedding_dim is None:
            embedding_dim = 100
        self.embedding_dim = embedding_dim
        self.word2em = load_glove(data_dir_path, embedding_dim)

    def encode_word(self, word):
        w = word.lower()
        if w in self.word2em:
            return self.word2em[w]
        else:
            return np.zeros(shape=(self.embedding_dim, ))

    def encode_docs(self, docs, max_allowed_doc_length=None):
        doc_count = len(docs)
        X = np.zeros(shape=(doc_count, self.embedding_dim))
        max_len = 0
        for doc in docs:
            max_len = max(max_len, len(doc.split(' ')))
        if max_allowed_doc_length is not None:
            max_len = min(max_len, max_allowed_doc_length)
        for i in range(0, doc_count):
            doc = docs[i]
            words = [w.lower() for w in doc.split(' ')]
            length = min(max_len, len(words))
            E = np.zeros(shape=(self.embedding_dim, max_len))
            for j in range(length):
                word = words[j]
                try:
                    E[:, j] = self.word2em[word]
                except KeyError:
                    pass
            X[i, :] = np.sum(E, axis=1)

        return X

    def encode_doc(self, doc, max_allowed_doc_length=None):
        words = [w.lower() for w in doc.split(' ')]
        max_len = len(words)
        if max_allowed_doc_length is not None:
            max_len = min(len(words), max_allowed_doc_length)
        E = np.zeros(shape=(self.embedding_dim, max_len))
        X = np.zeros(shape=(self.embedding_dim, ))
        for j in range(max_len):
            word = words[j]
            try:
                E[:, j] = self.word2em[word]
            except KeyError:
                pass
        X[:] = np.sum(E, axis=1)
        return X
