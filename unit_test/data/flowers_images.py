import unittest
import os
import sys
import logging
import mxnet as mx


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


class FlowersImagesUnitTest(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super(FlowersImagesUnitTest, self).__init__(methodName)
        logging.basicConfig(level=logging.DEBUG)

    def test_load_image_paths(self):

        data_dir_path = patch_path('../../demo/data/flowers/jpg')

        from mxnet_text_to_image.data.flowers_images import get_image_paths
        images_dict = get_image_paths(data_dir_path)
        self.assertEqual(8189, len(images_dict))

        for i, (image_id, image_path) in enumerate(images_dict.items()):
            if i == 4:
                break
            logging.info('image_id: %s image_path: %s', image_id, image_path)

    def test_get_image_features(self):
        data_dir_path = patch_path('../../demo/data/flowers/jpg')

        from mxnet_text_to_image.data.flowers_images import get_image_features
        features = get_image_features(data_dir_path, model_ctx=mx.gpu(0))
        self.assertTrue(0 in features)
        self.assertEqual(8189, len(features))

    def test_get_transformed_images(self):
        data_dir_path = patch_path('../../demo/data/flowers/jpg')

        from mxnet_text_to_image.data.flowers_images import get_transformed_images
        features = get_transformed_images(data_dir_path)
        self.assertEqual(8189, len(features))
        self.assertTrue(0 not in features)
        for i, (image_id, image) in enumerate(features.items()):
            if i == 2:
                break
            self.assertTupleEqual((3, 64, 64), image.shape)


if __name__ == '__main__':
    sys.path.append(patch_path('../..'))
    unittest.main()
