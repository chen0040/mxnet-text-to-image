import unittest
from mxnet_text_to_image.data.glove import glove_word2emb_300
import os
import sys
import logging


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), '..', path)


class LoadGloveTest(unittest.TestCase):

    def test_load_glove(self):
        logging.basicConfig(level=logging.DEBUG)

        sys.path.append(patch_path('..'))
        data_dir_path = patch_path('../demo/data/glove')
        emb = glove_word2emb_300(data_dir_path)
        self.assertEqual(400000, len(emb))


if __name__ == '__main__':
    unittest.main()


