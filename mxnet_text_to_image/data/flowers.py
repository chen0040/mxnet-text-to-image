from mxnet_text_to_image.data.flowers_images import get_image_features, get_transformed_images
from mxnet_text_to_image.data.flowers_texts import get_text_features
import mxnet as mx
import os


def get_data_iter(data_dir_path, glove_dir_path=None, max_sequence_length=-1, ctx=mx.cpu(), image_feature_extractor='vgg16',
                  image_width=224, image_height=224,
                  text_mode='add', batch_size=64):
    if glove_dir_path is None:
        glove_dir_path = os.path.join(os.path.dirname(data_dir_path), 'glove')

    if image_feature_extractor == 'vgg16':
        image_feats_dict = get_image_features(data_dir_path=os.path.join(data_dir_path, 'jpg'), model_ctx=ctx,
                                              image_width=image_width, image_height=image_height)
    else:
        image_feats_dict = get_transformed_images(data_dir_path=os.path.join(data_dir_path, 'jpg'),
                                                  image_width=image_width, image_height=image_height)
    text_feats, image_id_array = get_text_features(data_dir_path=os.path.join(data_dir_path, 'text_c10'),
                                                   glove_dir_path=glove_dir_path,
                                                   max_seq_length=max_sequence_length,
                                                   mode=text_mode)

    image_feats = list()
    for image_id in image_id_array:
        feats = image_feats_dict[image_id][0]
        image_feats.append(feats)

    return mx.io.NDArrayIter(data=[image_feats, text_feats], batch_size=batch_size, shuffle=True)
