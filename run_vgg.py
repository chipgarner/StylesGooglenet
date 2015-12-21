import styles
import setup_caffe_network as su
import models as ml
import get_layer_data as gd

iterator = [
    {
        'iter_n': 150,
        'start_sigma': 3.0,
        'end_sigma': 0.0,
        'start_step_size': 6.0,
        'end_step_size': 1.0
    },

]


layer = 'conv5_4'

su.SetupCaffe.gpu_on()
net = ml.NetModels.setup_vgg('../CommonCaffe/TrainedModels/')

style_data = gd.get_layers_data(net, 'ImagesIn/Class 67.jpg', layer)
subject_data = gd.get_layers_data(net, 'ImagesIn/smallwonder.jpg', layer)

stl = styles.Styles()
stl.content_plus_style(net, iterator, style_data, subject_data, layer)
