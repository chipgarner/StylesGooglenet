from unittest import TestCase
import get_layers_data as gd
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
    layers = dict(conv1_1=0, conv2_1=1, conv3_1=2, conv4_1=3, conv5_4=4)

    net = ml.NetModels.setup_vgg('PycharmProjects/ImageFromClass/')

    style_data = gd.get_layers_data(net, 'PycharmProjects/Styles/test/diScaled.jpg', layers)
    subject_data = gd.get_layers_data(net, 'PycharmProjects/Styles/test/us.jpg', layers)

    # for i in range(0, 5):
    #     style_data.append(np.load('PycharmProjects/Styles/test/style' + str(i) + '.npy'))
    #     subject_data.append(np.load('PycharmProjects/Styles/test/subject' + str(i) + '.npy'))

    def test_get_subject_data(self):
        assert self.subject_data[self.layers['conv5_4']].shape == (512, 14, 18)
        assert self.subject_data[self.layers['conv4_1']].shape == (512, 28, 36)
        assert self.subject_data[self.layers['conv3_1']].shape == (256, 55, 72)
        assert self.subject_data[self.layers['conv2_1']].shape == (128, 109, 144)
        assert self.subject_data[self.layers['conv1_1']].shape == (64, 217, 288)

    def test_get_style_data(self):
        assert self.style_data[4].shape == (512, 14, 18)
        assert self.style_data[3].shape == (512, 28, 36)
        assert self.style_data[2].shape == (256, 55, 72)
        assert self.style_data[1].shape == (128, 109, 144)
        assert self.style_data[0].shape == (64, 217, 288)

    def test_get_different_data(self):

        self.assertFalse(np.array_equal(self.subject_data[4], self.style_data[4]))
        self.assertFalse(np.array_equal(self.subject_data[3], self.style_data[3]))
        self.assertFalse(np.array_equal(self.subject_data[2], self.style_data[2]))
        self.assertFalse(np.array_equal(self.subject_data[1], self.style_data[1]))
        self.assertFalse(np.array_equal(self.subject_data[0], self.style_data[0]))

#        np.testing.assert_array_equal(subject_data[4], subject_data[4], "Should equal itself")
#        np.testing.assert_array_equal(subject_data[4], style_data[4], "Should not equal each other")
