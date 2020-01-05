import pickle
import glob
from scipy import linalg
import numpy as np
import PIL.Image
from os import sep, getcwd
def _unpickle(filename):
    """
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended the filename.
    """

    # Create full path for the file.

    print("Loading data: " + filename)

    with open(filename, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        print(file)
        data = pickle.load(file, encoding='bytes')

    return data


def load_cifar10_rgb():
    import keras.datasets.cifar10
    train, test = keras.datasets.cifar10.load_data()
    train_data, train_label = train[0], train[1]
    test_data, test_label = test[0], test[1]
    # convert to float(0,1)
    train_data = train_data.astype(np.float32, copy=False) / 255.
    test_data = test_data.astype(np.float32, copy=False) / 255.
    for i in range(train_data.shape[0]):
        for j in range(3):
            if j is not i % 3:
                train_data[i, :, :, j] = 0
    # for i in range(test_data.shape[0]):
    #    for j in range(3):
    #        if j is not i%3:
    #            test_data[i,:,:,j]=0
    return train_data, train_label, test_data, test_label

def load_cifar10():
    import keras.datasets.cifar10
    train,test=keras.datasets.cifar10.load_data()
    train_data,train_label=train[0],train[1]
    test_data,test_label=test[0],test[1]
    labeled_data, labeled_label = train_data[0:4000, :, :, :], train_label[0:4000]
    unlabeled_data = train_data[4000:, :, :, :]
    print(labeled_data.shape)
    # convert to float(0,1)
    labeled_data = (labeled_data.astype(np.float32, copy=False) / 255. - 0.5) * 2
    unlabeled_data = (unlabeled_data.astype(np.float32, copy=False) / 255. - 0.5) * 2
    test_data = (test_data.astype(np.float32, copy=False) / 255. - 0.5)*2
    return labeled_data,labeled_label,unlabeled_data,test_data,test_label


def load_cifar10_all():
    import keras.datasets.cifar10
    train, test = keras.datasets.cifar10.load_data()
    train_data, train_label = train[0], train[1]
    test_data, test_label = test[0], test[1]
    # convert to float(0,1)
    train_data = (train_data.astype(np.float32, copy=False) / 255. - 0.5) * 2
    test_data = (test_data.astype(np.float32, copy=False) / 255. - 0.5) * 2
    return train_data, train_label, test_data, test_label


def load_celebA():
    dataset_path = getcwd() + sep + 'dataset' + sep + 'celebA_64' + sep
    img_list = glob.glob(dataset_path + '*.jpg')
    print('loading images...')
    all_images = np.array([np.array(PIL.Image.open(fname)) for fname in img_list])
    print('normalizing images...')
    all_images = (all_images.astype(np.float32, copy=False) / 255. - 0.5) * 2
    return all_images, np.zeros(1), np.zeros(1), np.zeros(1)


def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    # initially normalized to 0~1 !
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    train_data = np.pad(np.reshape(train_data, (-1, 28, 28, 1)), pad_width=((0, 0), (2, 2), (2, 2), (0, 0)),
                        mode='constant', constant_values=0)
    test_data = np.pad(np.reshape(test_data, (-1, 28, 28, 1)), pad_width=((0, 0), (2, 2), (2, 2), (0, 0)),
                       mode='constant', constant_values=0)
    train_data = (train_data.astype(np.float32, copy=False) - 0.5) * 2
    test_data = (test_data.astype(np.float32, copy=False) - 0.5) * 2
    return train_data, train_label, test_data, test_label

def load_fashion():
    from tensorflow.examples.tutorials.mnist import input_data
    # initially normalized to 0~1 !
    mnist = input_data.read_data_sets("dataset/fashion", one_hot=True,source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
    train_data, train_label, test_data, test_label = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    train_data = np.pad(np.reshape(train_data, (-1, 28, 28, 1)), pad_width=((0, 0), (2, 2), (2, 2), (0, 0)),
                        mode='constant', constant_values=0)
    test_data = np.pad(np.reshape(test_data, (-1, 28, 28, 1)), pad_width=((0, 0), (2, 2), (2, 2), (0, 0)),
                       mode='constant', constant_values=0)
    train_data = (train_data.astype(np.float32, copy=False) - 0.5) * 2
    test_data = (test_data.astype(np.float32, copy=False) - 0.5) * 2
    return train_data, train_label, test_data, test_label

class ZCA(object):
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        if x is not None:
            self.fit(x)

    def fit(self, x):
        # it is defined over all color values, not only single one
        s = x.shape
        x = x.copy().reshape((s[0], np.prod(s[1:])))
        m = np.mean(x, axis=0)
        x = x.astype(np.float32) - m
        sigma = np.dot(x.T, x) / x.shape[0]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1. / np.sqrt(S + self.regularization)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S + self.regularization)))
        self.ZCA_mat = np.dot(tmp, U.T).astype(np.float32)
        self.inv_ZCA_mat = np.dot(tmp2, U.T).astype(np.float32)
        self.mean = m.astype(np.float32)

    def apply(self, x):
        s = x.shape
        if isinstance(x, np.ndarray):
            return (np.dot(x.reshape((s[0], np.prod(s[1:]))) - self.mean.reshape(1, -1), self.ZCA_mat)).reshape(s)
        else:
            raise NotImplementedError("Whitening only implemented for numpy arrays")

    def invert(self, x):
        s = x.shape
        if isinstance(x, np.ndarray):
            return (np.dot(x.reshape((s[0], np.prod(s[1:]))), self.inv_ZCA_mat + self.mean)).reshape(s)
        else:
            raise NotImplementedError("Whitening only implemented for numpy arrays")
if __name__=='__main__':
    import collections
    import numpy as np

    # labeled_data, labeled_label, unlabeled_data, test_data, test_label = load_cifar10()
    # print(labeled_label.shape)
    # print(collections.Counter(labeled_label.flatten()))
    all_images, _, _, _ = load_mnist()
    print(all_images.shape)
