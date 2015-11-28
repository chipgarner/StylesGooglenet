from unittest import TestCase
import numpy as np
import styles
import setup_caffe_network as su
import models as ml
import get_layer_data as gd

iterator = [
    {
        'iter_n': 5,
        'start_sigma': 0.0,
        'end_sigma': 0.0,
        'start_step_size': 6.0,
        'end_step_size': 1.0
    },

]

# These are functional tests, output images ae compared to saved images.
class TestStyles(TestCase):
    def test_content_plus_style(self):
        layer = 'inception_4c/3x3_reduce'

        su.SetupCaffe.gpu_on()
        net = ml.NetModels.setup_googlenet_model('../CommonCaffe/TrainedModels/')
        print 'loaded net'

        style_data = gd.get_layers_data(net, 'test/soft-grey.jpg', layer)
        subject_data = gd.get_layers_data(net, 'test/soft-grey.jpg', layer)
        print 'loaded data'

        stl = styles.Styles()
        vis = stl.content_plus_style(net, iterator, style_data, subject_data, layer)
        # np.save('test/test_result', vis)

        #  They should be the same iff the two input images are the same size
        assert style_data.shape == subject_data.shape

        test_vis = np.load('test/test_result.npy')
        assert np.allclose(vis, test_vis, 0.01)

    def test_functional_again(self):
        layer = 'inception_3b/output'

        su.SetupCaffe.gpu_on()
        net = ml.NetModels.setup_googlenet_model('../CommonCaffe/TrainedModels/')
        print 'loaded net'

        style_data = gd.get_layers_data(net, 'test/elephants2.jpg', layer)
        subject_data = gd.get_layers_data(net, 'test/soft-grey.jpg', layer)
        print 'loaded data'

        stl = styles.Styles()
        vis = stl.content_plus_style(net, iterator, style_data, subject_data, layer)
        # np.save('test/test2_result', vis)

        #  They should be the same iff the two input images are the same size
        self.assertFalse(style_data.shape == subject_data.shape)

        test_vis = np.load('test/test2_result.npy')
        assert np.allclose(vis, test_vis, 0.01)


