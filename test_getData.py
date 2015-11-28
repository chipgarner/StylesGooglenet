from unittest import TestCase
import get_layer_data as gd
import setup_caffe_network as su
import models as ml
import numpy as np

su.SetupCaffe.gpu_on()


def key_from_value(dictionary, val):
    for key, value in dictionary.iteritems():
        if value == val:
            return key
    return None


class TestGetData(TestCase):
    layer = 'conv3_1'

    net = ml.NetModels.setup_vgg('../CommonCaffe/TrainedModels/')  # Slow

    style_data = gd.get_layers_data(net, 'test/diScaled.jpg', layer)
    subject_data = gd.get_layers_data(net, 'test/us.jpg', layer)

    def test_get_subject_data(self):
        assert self.subject_data.shape == (256, 55, 72)

    def test_get_style_data(self):
        assert self.style_data.shape == (256, 55, 72)

    def test_get_different_data(self):
        # They should be different, if you don't use .copy they end up equal
        self.assertFalse(np.allclose(self.subject_data, self.style_data), .001)
