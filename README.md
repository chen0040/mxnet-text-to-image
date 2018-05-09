# mxnet-text-to-image

Text to Image translation using Generative Adversarial Network and MXNet

The current project implement the following DCGAN models that can perform text-to-image translation:

* [dcgan1.py](mxnet_text_to_image/library/dcgan1.py): In this model:
    * The generator takes in a randomly generated sequence of fixed length (default to 100) and the glove-encoded text features (of fixed length
    of 300), concatenates the two and pass them through a number of layers of Conv2DTranspose to produce 
    transformed images of shape (3, 224, 224) where 3 is the image channels and 224 are the width and height of
    the images. 
    * The discriminator takes image features (extracted using a pretrained VGG16 model) of shape (, 1000), which
    is concatenated with the glove-encoded text features and pass through Dense layers to produce a single output
    which is passed to SigmoidBinaryCrossEntropy loss function.
    * This model 
* [dcgan2.py](mxnet_text_to_image/library/dcgan2.py): In this model:
    * The generator takes in a randomly generated sequence of fixed length (default to 100) and the glove-encoded text features (of fixed length 
    of 300), concatenates the two and pass them through a number of layers of Conv2DTranspose to produce
    transformed images of shape (3, 64, 64) where 3 is the image channels and 64 are the width and height of 
    the images.
    * The discriminator takes the (3, 64, 64) transformed images and pass them through layers of Conv2D to produce
    a fixed-length numeric vector, which is then concatenated with the glove-encoded text features and pass through
    Dense layers to produce a single output, which is passed to SigmoidBinaryCrossEntropy loss function.
    
# Usage

The [demo] codes contains scripts on how to train the DCGAN models using the [flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
dataset. To train the [DCGan](mxnet_text_to_image/library/dcgan2.py) in [dcgan2.py](mxnet_text_to_image/library/dcgan2.py)
using the flowers dataset, run the following command:

```bash
python demo/dcgan2_train.py
```

Below shows the codes in the [demo/dcgan2_train.py](demo/dcgan2_train.py):

```python
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
                               image_width=64,
                               image_height=64,
                               batch_size=batch_size,
                               text_mode='add')

    gan = DCGan(model_ctx=ctx)
    gan.random_input_size = 100  # random input is 100, text input is 300

    gan.fit(train_data=train_data, model_dir_path=output_dir_path, epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    main()
```

After training, the trained models will be saved to the [demo/models](demo/models) folder with prefix "dcgan-v2-..."

To test the trained models in [demo/models], run the following command:

```bash
python demo/dcgan2_test.py
```

Below is the sample codes in [demo/dcgan2_test.py](demo/dcgan2_test.py):

```python
import os
import sys
import mxnet as mx
import logging


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))

    logging.basicConfig(level=logging.DEBUG)

    model_dir_path = patch_path('models')
    ctx = mx.gpu(0)

    from mxnet_text_to_image.library.dcgan2 import DCGan
    from mxnet_text_to_image.data.flowers_texts import load_texts

    gan = DCGan(model_ctx=ctx)
    gan.load_glove(glove_dir_path=patch_path('data/glove'))
    gan.load_model(model_dir_path=model_dir_path)

    texts = load_texts(patch_path('data/flowers/text_c10'))
    for i, (line, image_id) in enumerate(texts.items()):
        gan.generate(text_message=line, num_images=1, output_dir_path=patch_path('output'))


if __name__ == '__main__':
    main()

```
