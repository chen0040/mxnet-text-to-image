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
    from mxnet_text_to_image.data.flowers_images import get_transformed_images

    train_data = get_data_iter(data_dir_path=data_dir_path,
                               batch_size=batch_size,
                               text_mode='add')

    image_dict = get_transformed_images(data_dir_path=os.path.join(data_dir_path, 'jpg'),
                                        image_width=64, image_height=64)

    gan = DCGan(model_ctx=ctx)
    gan.random_input_size = 100  # random input is 100, text input is 300

    gan.fit(train_data=train_data, image_dict=image_dict, model_dir_path=output_dir_path,
            epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    main()
