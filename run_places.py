import styles
import setup_caffe_network as su
import models as ml
import get_layer_data as gd

iterator = [
    {
        'iter_n': 100,
        'start_sigma': 1.0,
        'end_sigma': 0.0,
        'start_step_size': 6.0,
        'end_step_size': 1.0
    },

]


layer = 'inception_4c/3x3_reduce'

su.SetupCaffe.gpu_on()
net = ml.NetModels.setup_places_model('../CommonCaffe/TrainedModels/')
print 'loaded net'

style_data = gd.get_layers_data(net, 'ImagesIn/Class 67.jpg', layer)
subject_data = gd.get_layers_data(net, 'ImagesIn/vvg.jpg', layer)
print 'loaded data'

stl = styles.Styles()
stl.content_plus_style(net, iterator, style_data, subject_data, layer)
