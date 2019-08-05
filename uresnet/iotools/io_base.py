from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .hdf5_loader import linear_train_loader
import numpy as np
import sys
import time
import h5py
import os

class io_base(object):

    def __init__(self, flags):

        if not flags.BATCH_SIZE % (flags.MINIBATCH_SIZE * len(flags.GPUS)) == 0:
            msg = 'BATCH_SIZE (%d) must be divisible by GPU count (%d) times MINIBATCH_SIZE(%d)'
            msg = msg % (flags.BATCH_SIZE, len(flags.GPUS), flags.MINIBATCH_SIZE)
            print(msg)
            raise ValueError
        
        self._minibatch_per_step = flags.MINIBATCH_SIZE * len(flags.GPUS)
        self._minibatch_per_gpu  = flags.MINIBATCH_SIZE
        self._num_entries  = -1
        self._num_channels = -1
        self._flags = flags
        self._blob = {}
        self.tspent_io = 0
        self.tspent_sum_io = 0
        self.loader = self._loader(flags.INPUT_FILE[0])
        
    def blob(self):
        return self._blob

    def batch_per_step(self):
        return self._minibatch_per_step

    def batch_per_gpu(self):
        return self._minibatch_per_gpu

    def num_entries(self):
        return self._num_entries

    def num_channels(self):
        return self._num_channels

    def initialize(self):
        raise NotImplementedError

    def set_index_start(self,idx):
        raise NotImplementedError

    def start_threads(self):
        raise NotImplementedError

    def stop_threads(self):
        raise NotImplementedError

    def _loader(self, directory):
        tstart = time.time()
        print('directory', directory)
        print(os.listdir(directory))
        for f in os.listdir(directory):
            filename = directory + f
            if not os.path.isfile(filename):
                print(filename, "not a file")
                continue
            elif filename[-5:] != '.hdf5':
                continue
            print('good filename', filename)

            for d, c, f, l, nepe in linear_train_loader(filename, self._flags.BATCH_SIZE):
                self._num_entries = len(nepe)
                idx = [np.array(nepe)]
                blob = {}
                blob['voxels'] = [c]
                blob['data'] = [np.hstack((c, f))]
                blob['feature'] = [f]
                blob['label'] = [l]
                self.tspent_io = time.time() - tstart
                tstart = time.time()
                self.tspent_sum_io += self.tspent_io
                yield idx, blob

    def next(self, buffer_id=-1, release=True):
        return next(self.loader)
        
    def _next(self,buffer_id=-1,release=True):
        raise NotImplementedError
    
    def store_segment(self, idx, data, softmax):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError
