import os
import sys
import mxnet as mx
import logging


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))

    logging.basicConfig(level=logging.DEBUG)

    data_dir_path = patch_path('data/flowers')
    output_dir_path = patch_path('models')
    batch_size = 64
    epochs = 100
    ctx = mx.gpu(0)

    from mxnet_text_to_image.library.dcgan2 import DCGan
    from mxnet_text_to_image.data.flowers import get_data_iter

    train_data = get_data_iter(data_dir_path=data_dir_path,
                               ctx=mx.cpu(),
                               image_feature_extractor=None,
                               image_width=56,
                               image_height=56,
                               batch_size=batch_size,
                               text_mode='add')

    gan = DCGan(model_ctx=ctx)
    gan.random_input_size = 100  # random input is 100, text input is 300

    gan.fit(train_data=train_data, model_dir_path=output_dir_path, epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    main()
