from mxnet_text_to_image.data.flowers_images import get_image_features, get_transformed_images
from mxnet_text_to_image.data.flowers_texts import get_text_features
import mxnet as mx
from mxnet import nd
import os
import numpy as np


def get_data_iter(data_dir_path, glove_dir_path=None, max_sequence_length=-1,
                  text_mode='add', batch_size=64):
    if glove_dir_path is None:
        glove_dir_path = os.path.join(os.path.dirname(data_dir_path), 'glove')

    text_feats, image_id_array = get_text_features(data_dir_path=os.path.join(data_dir_path, 'text_c10'),
                                                   glove_dir_path=glove_dir_path,
                                                   max_seq_length=max_sequence_length,
                                                   mode=text_mode)

    return mx.io.NDArrayIter(data=[nd.array(image_id_array, ctx=mx.cpu()), text_feats], batch_size=batch_size, shuffle=True)
