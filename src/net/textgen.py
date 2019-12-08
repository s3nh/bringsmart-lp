import numpy as np 
import os 



IMAGES_PATH = 'data'
class DataPreparation(self, img_dir, batch_size, img_size, num_channels = 3, label_len = 5 ):
    self.img_dir = img_dir
    self.batch_size = batch_size 
    self.img_size = img_size
    self.num_channels = num_channels
    self.label_len = label_len
    self.img_w, self.img_h = img_size

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
            start = self._next_index
            end = self._next_index + batch_size
            self._num_epoches += 1
        else:
            self._next_index = end
        images = np.zeros([batch_size, self.img_h, self.img_w, self.num_channels])
    
        for j, i in enumerate(range(start, end)):
            fname = self.filenames[i]
            img = cv2.imread(os.path.join(self.img_dir, fname))
            img = cv2.resize(img, (self.img_w, self.img_h), interpolation = cv2.INTER_CUBIC )
            images[j, ...] = img
        images = np.transpose( images, axes=[0,2,1, 3])
        labels = self.labels[start:end, ... ]
        targets = [np.asarray(i) for i in labels]
        sparse_labels = sparse_tuple_from(targets)





