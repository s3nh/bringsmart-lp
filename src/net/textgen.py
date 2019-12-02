import numpy as np 
import os 



IMAGES_PATH = 'data'
class DataPreparation(self, img_dir, batch_size, img_size, num_channels = 3, label_len = 5 ):
    self.img_dir = img_dir
    self.batch_size = batch_size 
    self.img_size = img_size
    self.num_channels = num_channels
    self.label_len = label_len

    # Create empty objects 

    self._num_examples = 0
    self._next_index = 0 
    self._num_epochs = 0 
    self.filenames = []
    self.labels = []
    self.init()

    def init(self):
        self.labels = []
        # List all files 
        fs = os.listdir(self.img_dir)
        for file in fs:
            _labels = file[:-3]
            self.filenames.append(file)
            self.labels.append(_labels)
        self.labels = np.float32(self.labels)

    def data_shuffle(self):
        _ix = np.arange(self._num_examples)
        np.random.shuffle(_ix)
        self.filenames = [self.filenames[ix] for ix in _ix]
        self._labels = self._labels[_ix]
        return self.filenames , self._labels

    def next_batch(self):
        batch_size = self.batch_size
        end = self._next_index + batch_size
        if end > self._num_examples:
            self._next_index = 0
            start 