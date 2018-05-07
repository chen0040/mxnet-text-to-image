import os
import numpy as np
from mxnet_text_to_image.utils.image_utils import Vgg16FeatureExtractor, transform_image
import logging
import mxnet as mx

def get_image_paths(data_dir_path):
    result = dict()
    for root_dir, _, files in os.walk(data_dir_path):
        for fname in files:
            if fname.endswith('.jpg'):
                image_name = fname.replace('.jpg', '')
                image_id = int(image_name.replace('image_', ''))
                fpath = os.path.join(root_dir, fname)
                result[image_id] = fpath

    return result


def get_image_features(data_dir_path, model_ctx=mx.cpu(), image_width=224, image_height=224):
    features = dict()
    features_path = os.path.join(os.path.dirname(data_dir_path), 'flower_image_feats.npy')
    if os.path.exists(features_path):
        logging.debug('loading image features from %s', features_path)
        features = np.load(features_path).item()

    image_paths_dict = get_image_paths(data_dir_path)

    fe = Vgg16FeatureExtractor(model_ctx)

    total_images = len(image_paths_dict)

    changed = False
    for i, (image_id, image_path) in enumerate(image_paths_dict.items()):
        if image_id in features:
            continue
        feats = fe.extract_image_features(image_path, image_width=image_width, image_height=image_height).asnumpy()
        features[image_id] = feats[0]
        changed = True
        if i % 500 == 0:
            logging.debug('Has extracted features from %d images out of %d images (%.2f %%)', i+1, total_images, (i+1) * 100 / total_images)
            if changed:
                np.save(features_path, features)
                changed = False

    if changed:
        np.save(features_path, features)
    return features


def get_transformed_images(data_dir_path, image_width=64, image_height=64):
    features = dict()
    features_path = os.path.join(os.path.dirname(data_dir_path), 'flower_transformed_images.npy')
    if os.path.exists(features_path):
        logging.debug('loading transformed images from %s', features_path)
        features = np.load(features_path).item()

    image_paths_dict = get_image_paths(data_dir_path)

    total_images = len(image_paths_dict)

    changed = False
    for i, (image_id, image_path) in enumerate(image_paths_dict.items()):
        if image_id in features:
            continue
        features[image_id] = transform_image(image_path, image_width=image_width, image_height=image_height).asnumpy()
        changed = True
        if i % 1000 == 0:
            logging.debug('Has transformed %d images out of %d images (%.2f %%)', i+1, total_images, (i+1) * 100 / total_images)
            if changed:
                np.save(features_path, features)
                changed = False

    if changed:
        np.save(features_path, features)
    return features




