import images


def k_from_v(dictionary, val):
    for key, value in dictionary.iteritems():
        if value == val:
            return key
    return None


def get_layers_data(net, img_path, layer):
    im = images.Images()
    im.load_image(net, img_path)
    print 'loaded image'

    net.forward(end=layer)

    print 'went forward'
    layer_data = net.blobs[layer].data[0].copy()

    return layer_data
