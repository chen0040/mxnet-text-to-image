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
    batch_size = 8
    epochs = 100
    ctx = mx.gpu(0)

    from mxnet_text_to_image.library.dcgan1 import DCGan
    from mxnet_text_to_image.data.flowers import get_data_iter
    from mxnet_text_to_image.data.flowers_images import get_image_features

    train_data = get_data_iter(data_dir_path=data_dir_path,
                               batch_size=batch_size,
                               text_mode='add')

    image_feats_dict = get_image_features(data_dir_path=os.path.join(data_dir_path, 'jpg'), model_ctx=ctx,
                                          image_width=224, image_height=224)

    gan = DCGan(model_ctx=ctx)
    gan.random_input_size = 100  # random input is 100, text input is 300

    gan.fit(train_data=train_data, image_feats_dict=image_feats_dict, model_dir_path=output_dir_path,
            epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    main()
