import numpy as np
import nltk

def pad_sequence(seq, max_sequence_length, padding):
    seq = np.array(seq)
    vec = np.zeros(shape=(max_sequence_length, *seq.shape[1:]))
    if padding == 'left':
        start_index = 0
        if len(seq) < max_sequence_length:
            start_index = max_sequence_length - len(seq)
        vec[start_index:] = seq[:min(len(seq), max_sequence_length)]
    else:
        length = min(len(seq), max_sequence_length)
        vec[:length] = seq[:length]
    return vec


def pad_sequences(seq_list, max_sequence_length=-1, padding='left'):
    seq_count = len(seq_list)

    if max_sequence_length == -1:
        max_sequence_length = max([len(seq) for seq in seq_list])

    first_item = np.array(seq_list[0])
    matrix = np.zeros(shape=(seq_count, max_sequence_length, *first_item.shape[1:]))

    for i, seq in enumerate(seq_list):
        matrix[i] = pad_sequence(seq, max_sequence_length, padding)

    return matrix


def word_tokenize(sentence):
    return nltk.word_tokenize(sentence)





