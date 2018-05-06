import os
import logging
from mxnet_text_to_image.utils.glove import glove_word2emb_300
from mxnet_text_to_image.utils.text_utils import word_tokenize, pad_sequence
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


def get_text_features(data_dir_path, glove_dir_path=None, max_seq_length=-1, mode='add'):
    if mode == 'concat':
        features_path = os.path.join(os.path.dirname(data_dir_path), 'flower_text_feats_' + mode + '_'
                                     + str(max_seq_length) + '.npy')
    else:
        features_path = os.path.join(os.path.dirname(data_dir_path), 'flower_text_feats_' + mode + '.npy')
    if os.path.exists(features_path):
        logging.debug('loading text features from %s', features_path)
        return np.load(features_path), np.load(features_path[:len(features_path)-4] + '_mapping.npy')

    if glove_dir_path is None:
        glove_dir_path = os.path.join(os.path.dirname(os.path.dirname(data_dir_path)), 'glove')
    emb = glove_word2emb_300(glove_dir_path)
    texts = load_texts(data_dir_path)
    result = list()
    mapping = list()
    total_images = len(texts)

    if mode == 'concat' or mode == 'concat_int':
        temp = max_seq_length
        for i, (image_id, lines) in enumerate(texts.items()):
            for line in lines:
                words = word_tokenize(line.lower())
                max_seq_length = max(len(words), max_seq_length)
        if temp > 0:
            max_seq_length = min(temp, max_seq_length)
    elif mode == 'add':
        max_seq_length = -1

    logging.debug('max sequence length: %d', max_seq_length)

    for i, (image_id, lines) in enumerate(texts.items()):
        for k, line in enumerate(lines):
            words = word_tokenize(line.lower())
            if mode == 'concat':
                temp = list()
                for word in words:
                    if word in emb:
                        temp.append(emb[word])
                    else:
                        temp.append(np.zeros(shape=300))
                result.append(pad_sequence(temp, max_seq_length))
            else:
                encoded = np.zeros(shape=(len(words), 300))
                for j, word in enumerate(words):
                    if word in emb:
                        encoded[j, :] = emb[word]

                encoded = np.sum(encoded, axis=0).reshape(300)
                result.append(encoded)
            mapping.append(image_id)
        if i % 100 == 0:
            logging.debug('Has extracted text features from %d images out of %d images (%.2f %%)', i + 1, total_images,
                          (i + 1) * 100 / total_images)

    result = np.array(result)
    mapping = np.array(mapping)
    np.save(features_path, result)
    np.save(features_path[:len(features_path)-4] + '_mapping.npy', mapping)
    return result, mapping
