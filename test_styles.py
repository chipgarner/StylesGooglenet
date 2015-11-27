from unittest import TestCase
import numpy as np
import styles
import setup_caffe_network as su
import models as ml
import get_layers_data as gd

iterator = [
    {
        'iter_n': 5,
        'start_sigma':1.0,
        'end_sigma':0.0,
        'start_step_size':6.0,
        'end_step_size':1.0
    },

]


class TestStyles(TestCase):
    def test_content_plus_style(self):
        # VGG layers layers = dict(conv1_1=0, conv2_1=1, conv3_1=2, conv4_1=3, conv5_4=4)
        layers = 'inception_4c/3x3_reduce'

        su.SetupCaffe.gpu_on()
        net = ml.NetModels.setup_googlenet_model('Convolutions/')
        print 'loaded net'

        style_data = gd.get_layers_data(net, 'PycharmProjects/GooglenetStyles/ImagesIn/soft-grey.jpg', layers)
        subject_data = gd.get_layers_data(net, 'PycharmProjects/GooglenetStyles/ImagesIn/soft-grey.jpg', layers)
        print 'loaded data'

        # for i in range(1, 5):
        #     np.save('PycharmProjects/Styles/test/style' + str(i), style_data[i])
        #     np.save('PycharmProjects/Styles/test/subject' + str(i), subject_data[i])

        stl = styles.Styles()
        stl.content_plus_style(net, iterator, style_data, subject_data, layers)

        #  They should be the same if the two input images are the same size
        assert style_data[0].shape == subject_data[0].shape
        # self.assertFalse(np.array_equal(subject_data[4], style_data[4]))

