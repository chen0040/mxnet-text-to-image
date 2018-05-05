import os
import logging
from mxnet_text_to_image.utils.glove import glove_word2emb_300
from mxnet_text_to_image.utils.text_utils import word_tokenize
import numpy as np


def load_text_files(data_dir_path):
    result = dict()
    for root_dir, sub_dirs, files in os.walk(data_dir_path):
        for fname in files:
            if fname.endswith('.txt'):
                result[fname.replace('.txt', '')] = os.path.join(root_dir, fname)

    return result


def load_texts(data_dir_path):
    result = dict()
    image_id_2_text_file_paths = load_text_files(data_dir_path)
    total_files = len(image_id_2_text_file_paths)
    for i, (image_id, text_file_path) in enumerate(image_id_2_text_file_paths.items()):
        with open(text_file_path, 'r') as f:
            lines = list()
            for line in f:
                lines.append(line)
            result[image_id] = lines
        if i % 500 == 0:
            logging.debug('Number of text files loaded so far: %d / %d', i + 1, total_files)
    return result


def get_text_features(data_dir_path, glove_dir_path=None):
    features_path = os.path.join(os.path.dirname(data_dir_path), 'flower_text_feats.npy')
    if os.path.exists(features_path):
        logging.debug('loading text features from %s', features_path)
        return np.load(features_path).item()

    if glove_dir_path is None:
        glove_dir_path = os.path.join(os.path.dirname(os.path.dirname(data_dir_path)), 'glove')
    emb = glove_word2emb_300(glove_dir_path)
    texts = load_texts(data_dir_path)
    result = list()
    total_images = len(texts)
    for i, (image_id, lines) in enumerate(texts.items()):
        for line in lines:
            words = word_tokenize(line.lower())
            encoded = list()
            for word in words:
                if word in emb:
                    em = emb[word]
                else:
                    em = np.zeros(shape=300)
                encoded.append(em)
            result.append((image_id, encoded))
        if i % 100 == 0:
            logging.debug('Has extracted text features from %d images out of %d images (%.2f %%)', i + 1, total_images,
                          (i + 1) * 100 / total_images)

    np.save(features_path, result)
    return result
