import os
from PIL import Image
import numpy as np
from joblib import Parallel, delayed


def load_one_image(filepath, new_size=None):
    """new_size is as in numpy: (# lines, # columns, # channels)."""
    img = Image.open(filepath)
    if new_size is not None:
        # resize image
        if new_size[0] != img.size[1] or new_size[1] != img.size[0]:   # height/width are inverted between numpy as PIL
            img = img.resize((new_size[1], new_size[0]), Image.ANTIALIAS)
        # turn image to black and white
        if new_size[2] == 1 and img.mode == 'RGB':
            img = img.convert('L')
    if len(np.asarray(img).shape) == 2:
        # for black and white pictures we need to add one channel to have a 3d array
        old_shape = np.asarray(img).shape
        new_shape = (old_shape[0], old_shape[1], 1)
        return np.asarray(img).reshape(new_shape)
    else:
        return np.asarray(img)


class DataSet:
    def __init__(self, folder, new_size=None, substract_mean=False, subsample_size=None, test=None):
        # paths
        self.folder = folder   # must contain one 'train' and one 'test' folder
        self.new_size = new_size
        self.substract_mean = substract_mean
        # for debugging
        self.subsample_size = subsample_size
        # loading training data data
        self.labels, self.train_files, self.prediction_files = self.load_paths()
        self.images = self.load_images_from_list(self.train_files)
        self.images_mean = self.get_images_mean()
        self.num_examples = len(self.train_files)
        # for train batch
        self.indexes = np.arange(self.num_examples)
        self.index_in_epoch = 0
        self.epochs_completed = -1
        # same thing for test data
        self.test_labels = None
        self.test_images = None
        self.test_files = self.prediction_files.copy()
        if test is not None:
            self.test_files = test.image.values
            self.test_labels = test.label.values
            self.test_images = self.load_images_from_list(self.test_files)
        self.test_num_examples = len(self.test_files)
        self.test_indexes = np.arange(len(self.test_files))
        self.test_index_in_epoch = 0
        self.test_epochs_completed = -1
        # same thing for prediction data
        self.prediction_num_examples = len(self.prediction_files)
        self.prediction_indexes = np.arange(self.prediction_num_examples)
        self.prediction_index_in_epoch = 0

    def load_paths(self):
        """Load the files paths and categories for train and test."""
        train_folder = self.folder + '/train'
        train_files = []
        labels = []
        for subfolder in [s for s in os.listdir(train_folder) if s != '.DS_Store']:
            path_ = train_folder + '/' + subfolder
            train_files += [os.path.join(path_, f) for f in os.listdir(path_) if f != '.DS_Store']
            labels += [int(subfolder.replace('c', '')) for f in os.listdir(train_folder + '/' + subfolder) if f != '.DS_Store']
        test_files = [os.path.join(self.folder + '/test', f) for f in os.listdir(self.folder + '/test')]
        if self.subsample_size is not None:
            idx = np.random.choice(np.arange(len(train_files)), self.subsample_size, replace=False)
            return np.array(labels)[idx], np.array(train_files)[idx], np.array(test_files)[idx]
        else:
            return np.array(labels), np.array(train_files), np.array(test_files)

    def load_images_from_list(self, file_list):
        """Load some images in parallel given their path."""
        images = Parallel(n_jobs=-1)(delayed(load_one_image)(file_, self.new_size) for file_ in file_list)
        return np.array(images)

    def get_images_mean(self, mean_pixel=False):
        """Calculates the mean image or mean pixel"""
        if mean_pixel:
            mean_pixel = self.images.mean(axis=0).mean(axis=0).mean(axis=0)
            mean_image = np.zeros((128, 128, 3))
            mean_image[:] = mean_pixel
            return mean_image
        else:
            return self.images.mean(axis=0)

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples or self.epochs_completed == -1:
            # finished epoch
            self.epochs_completed += 1
            # shuffle the index
            np.random.shuffle(self.indexes)
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        if self.substract_mean:
            return (self.images[self.indexes[start:end]] - self.images_mean) / 255. - 0.5,\
                np.eye(10)[self.labels[self.indexes[start:end]]]
        else:
            return self.images[self.indexes[start:end]] / 255. - 0.5,\
                np.eye(10)[self.labels[self.indexes[start:end]]]

    def next_test_batch(self, batch_size):
        """Return the next `batch_size` examples from the test data set."""
        # get indexes
        start = self.test_index_in_epoch
        self.test_index_in_epoch += batch_size
        if self.test_index_in_epoch > self.test_num_examples or self.test_epochs_completed == -1:
            # finished epoch
            self.test_epochs_completed += 1
            # shuffle the index
            np.random.shuffle(self.test_indexes)
            # start next epoch
            start = 0
            self.test_index_in_epoch = batch_size
            assert batch_size <= self.test_num_examples
        end = self.test_index_in_epoch
        # load images
        if self.substract_mean:
            return (self.test_images[self.test_indexes[start:end]] - self.images_mean) / 255. - 0.5,\
                np.eye(10)[self.test_labels[self.test_indexes[start:end]]]
        else:
            return self.test_images[self.test_indexes[start:end]] / 255. - 0.5,\
                np.eye(10)[self.test_labels[self.test_indexes[start:end]]]

    def next_prediction_batch(self, batch_size):
        """Return the next `batch_size` examples from the test data set."""
        # get indexes
        start = self.prediction_index_in_epoch
        self.prediction_index_in_epoch += batch_size
        end = min(self.prediction_index_in_epoch, self.prediction_num_examples)
        # load images
        prediction_images = self.load_images_from_list(self.prediction_files[start:end])
        if self.substract_mean:
            return (prediction_images - self.images_mean) / 255. - 0.5
        else:
            return prediction_images / 255. - 0.5
