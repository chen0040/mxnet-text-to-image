import unittest
from mxnet_text_to_image.data.flowers_texts import load_texts
import os
import sys
import logging

def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


class FlowersTextTest(unittest.TestCase):

    def test_load_texts(self):
        sys.path.append(patch_path('../..'))
        data_dir_path = patch_path('../../demo/data/flowers/text_c10')

        logging.basicConfig(level=logging.DEBUG)
        texts = load_texts(data_dir_path)
        self.assertEqual(8189, len(texts))
        for i, (image_id, lines) in enumerate(texts.items()):
            if i == 4:
                break
            for line in lines:
                logging.info('image: %s line: %s', image_id, line)


if __name__ == '__main__':
    unittest.main()
