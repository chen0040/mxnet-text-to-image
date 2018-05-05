import unittest
import os
import sys
import logging

def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


class FlowersTextTest(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super(FlowersTextTest, self).__init__(methodName)

    def test_load_texts(self):
        data_dir_path = patch_path('../../demo/data/flowers/text_c10')

        from mxnet_text_to_image.data.flowers_texts import load_texts
        texts = load_texts(data_dir_path)
        self.assertEqual(8189, len(texts))
        for i, (image_id, lines) in enumerate(texts.items()):
            if i == 4:
                break
            for line in lines:
                logging.info('image: %s line: %s', image_id, line)

    def test_get_text_features(self):

        from mxnet_text_to_image.data.flowers_texts import get_text_features
        feats = get_text_features(data_dir_path=patch_path('../../demo/data/flowers/text_c10'),
                                  glove_dir_path=patch_path('../../demo/data/glove'))




if __name__ == '__main__':
    sys.path.append(patch_path('../..'))
    unittest.main()
