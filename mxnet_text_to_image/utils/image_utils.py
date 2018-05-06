import numpy as np
from mxnet import nd, image
from mxnet.gluon.model_zoo import vision as models
import mxnet as mx
from PIL import Image

rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def transform(data):
    data = data.transpose((2, 0, 1))
    return (data.astype(np.float32) / 255 - rgb_mean) / rgb_std


def inverted_transform(img):
    return ((img * rgb_std + rgb_mean) * 255).transpose(1, 2, 0)

def load_vgg16_image(img_path):
    x = image.imread(img_path)
    x = image.resize_short(x, 256)
    x, _ = image.center_crop(x, (224, 224))
    return x


def save_image(img_data, save_to_file):
    Image.fromarray(img_data).save(save_to_file)


class Vgg16FeatureExtractor(object):

    def __init__(self, model_ctx=mx.cpu()):
        self.model_ctx = model_ctx
        self.image_net = models.vgg16(pretrained=True)
        self.image_net.collect_params().reset_ctx(ctx=model_ctx)

    def extract_image_features(self, image_path):
        img = load_vgg16_image(image_path)
        img = transform(img).expand_dims(axis=0)
        img = img.as_in_context(self.model_ctx)
        return self.image_net(img)
