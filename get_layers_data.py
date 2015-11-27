import images


def k_from_v(dictionary, val):
    for key, value in dictionary.iteritems():
        if value == val:
            return key
    return None


def get_layers_data(net, img_path, layers):
    im = images.Images()
    im.load_image(net, img_path)
    print 'loaded image'

    net.forward(end=layers)

    print 'went forward'
    layers_data = []

    # for i in range(0, len(layers)):
    layers_data.append(net.blobs[layers].data[0].copy())

    return layers_data
