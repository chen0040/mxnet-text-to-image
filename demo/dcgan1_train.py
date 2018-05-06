import os
import sys
import mxnet as mx


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))

    data_dir_path = patch_path('data/flowers')
    output_dir_path = patch_path('models')
    batch_size=64
    epochs=100

    from mxnet_text_to_image.library.dcgan1 import DCGan
    from mxnet_text_to_image.data.flowers import get_data_iter

    train_data = get_data_iter(data_dir_path=data_dir_path,
                               ctx=mx.gpu(0),
                               batch_size=batch_size,
                               text_mode='add')

    gan = DCGan(model_ctx=mx.gpu(0), data_ctx=mx.gpu(0))

    gan.fit(train_data=train_data, model_dir_path=output_dir_path, epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    main()
