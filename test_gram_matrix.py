from unittest import TestCase
import numpy as np
import gramian


class TestGram_matrix(TestCase):
    def test_gram_matrix(self):
        style_data = []
        for i in range(0, 5):
            style_data.append(np.load('PycharmProjects/Styles/test/style' + str(i) + '.npy'))

        print str(style_data[4].shape)
        gram = gramian.gram_matrix(style_data[4])

        assert gram.shape == (512, 512)
        self.assertAlmostEqual(gram[113, 113], 82854.664, 3)
        self.assertAlmostEqual(style_data[4][113, 1, 1], -36.442001, 5)

       # assert np.array_equal(style_data[1], gram)