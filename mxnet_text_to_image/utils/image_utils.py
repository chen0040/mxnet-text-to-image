import numpy as np
from mxnet import nd, image
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.utils import download
import mxnet as mx

def transform(data):
    data = data.transpose((2, 0, 1)).expand_dims(axis=0)
    rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
    rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
    return (data.astype(np.float32) / 255 - rgb_mean) / rgb_std


def load_vgg16_image(img_path):
    x = image.imread(img_path)
    x = image.resize_short(x, 256)
    x, _ = image.center_crop(x, (224, 224))
    return x


class Vgg16FeatureExtractor(object):

    def __init__(self, model_ctx=mx.cpu()):
        self.model_ctx = model_ctx
        self.image_net = models.vgg16(pretrained=True)
        self.image_net.collect_params().reset_ctx(ctx=model_ctx)

    def extract_image_features(self, image_path):
        img = load_vgg16_image(image_path)
        img = transform(img)
        img = img.as_in_context(self.model_ctx)
        return self.image_net(img)
