import PIL.Image
import cudarray as ca
import numpy as np
import scipy.ndimage as nd
import display
import images


class Styles:
    def __init__(self):
        pass

    def __save_result(self, index, image_path, end, vis):
        pimg = PIL.Image.fromarray(vis)

        # get the image name from the path
        txt = image_path.split('/')
        nm = txt[len(txt) - 1].split('.')
        name = nm[0]

        f = "frames/" + str(index) + '_' + name + '_' + end.replace('/', '-') + ".jpg"
        pimg.save(f, 'jpeg')
        print f

    def __blur(self, img, sigma):
        if sigma > 0:
            img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
            img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
            img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
        return img

    def __objective_L2(self, dst):
        dst.diff[:] = dst.data

    def __objective_L2_subject(self, dst, subject, style):
        dst.diff[:] = subject

    @staticmethod
    def __gram_matrix(v):
        n_channels = v.shape[0]
        feats = np.reshape(v, (n_channels, -1))
        gram = np.dot(feats, feats.T)
        return gram

    # From DeepDream ipython
    def __objective_guide(self, dst, subject, style):
        x = subject
        y = style
        ch = x.shape[0]
        x = np.reshape(x, (ch, -1))
        y = np.reshape(y, (ch, -1))
        A = np.dot(x.T, y)  # compute the matrix of dot-products with guide features

        dst.diff[0].reshape(ch, -1)[:] = y[:, A.argmax(1)]  # select ones that match best
        dst.diff[:] += dst.data  # this makes more network stuff appear

    # def __objective(self, dst, subject, style):
    #     x = dst.data[0].copy()
    #     y = style
    #     ch = x.shape[0]
    #     x = x.reshape(ch, -1)
    #     y = y.reshape(ch, -1)
    #
    #     xg = np.dot(x.T, x)  #Gramian
    #     yg = np.dot(y.T, y)
    #     A = xg - yg
    #     # A = x.T.dot(y)  # compute the matrix of dot-products with guide features
    #     dst.diff[0].reshape(ch, -1)[:] = y[:, A.argmax(1)]  # select ones that match best
    #
    #     dst.diff[:] += dst.data

    def __gradient_ascent(self, src, sigma, step_size):
        # src = net.blobs['data']  # input image is stored in Net's 'data' blob
        g = src.diff[0]

        # apply normalized ascent step to the input image
        src.data[:] += step_size / np.abs(g).mean() * g

        src.data[0] = self.__blur(src.data[0], sigma)

    def __take_steps(self, net, sigma, step_size, style_data, subject_data, layer):
        top = layer

        net.forward(end=top)

        self.__objective_guide(net.blobs[top], subject_data, style_data)

        bottom = None
        net.backward(start=top, end=bottom)
        self.__gradient_ascent(net.blobs['data'], sigma, step_size)

    def content_plus_style(self, net, iterator, style_data, subject_data, layer):

        im = images.Images()

        for e, o in enumerate(iterator):

            for i in xrange(o['iter_n']):
                sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
                step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']

                self.__take_steps(net, sigma, step_size, style_data, subject_data, layer)

                vis = im.visualize_src(net)
                self.__save_result(i, '', layer, vis)

            display.Display().showResultPIL(vis)
            return vis

