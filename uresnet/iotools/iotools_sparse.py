from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import threading
import time
from uresnet.iotools.io_base import io_base

def threadio_func(io_handle, thread_id):
    """
    Structure of returned blob:
        - voxels = [(N, 4)] * batch size
        - feature = [(N, 1)] * batch_per_step
        - data = [(N, 5)] * batch_per_step
        - label = [(N, 1)] * batch size
    where N = total number of points across minibatch_size events
    """
    num_gpus = len(io_handle._flags.GPUS)
    batch_per_step = io_handle.batch_per_step()
    batch_per_gpu = io_handle.batch_per_gpu()
    while 1:
        time.sleep(0.000001)
        while not io_handle._locks[thread_id]:
            idx_v     = []
            voxel_v   = []
            feature_v = []
            new_idx_v = []
            # label_v   = []
            blob = {}
            for key, val in io_handle.blob().iteritems():
                blob[key] = []
            if io_handle._flags.SHUFFLE:
                idx_v = np.random.random([batch_per_step])*io_handle.num_entries()
                idx_v = idx_v.astype(np.int32)
                # for key, val in io_handle.blob().iteritems():
                #     blob[key] = val  # fixme start, val?
            else:
                start = io_handle._start_idx[thread_id]
                end   = start + batch_per_step
                if end < io_handle.num_entries():
                    idx_v = np.arange(start,end)
                    # for key, val in io_handle.blob().iteritems():
                    #     blob[key] = val[start:end]
                else:
                    idx_v = np.arange(start, io_handle.num_entries())
                    idx_v = np.concatenate([idx_v,np.arange(0,end-io_handle.num_entries())])
                    # for key, val in io_handle.blob().iteritems():
                    #     blob[key] = val[start:] + val[0:end-io_handle.num_entries()]
                next_start = start + len(io_handle._threads) * batch_per_step
                if next_start >= io_handle.num_entries():
                    next_start -= io_handle.num_entries()
                io_handle._start_idx[thread_id] = next_start

            for i in range(num_gpus):
                voxel_v.append([])
                feature_v.append([])
                new_idx_v.append([])
                for key in io_handle._flags.DATA_KEYS:
                    blob[key].append([])

            for data_id, idx in enumerate(idx_v):
                voxel  = io_handle.blob()['voxels'][idx]
                new_id = int(data_id / batch_per_gpu)
                voxel_v[new_id].append(np.pad(voxel, [(0,0),(0,1)],'constant',constant_values=data_id))
                feature_v[new_id].append(io_handle.blob()['feature'][idx])
                new_idx_v[new_id].append(idx)
                for key in io_handle._flags.DATA_KEYS:
                    blob[key][new_id].append(io_handle.blob()[key][idx])
                # if len(io_handle._label):
                #     label_v.append(io_handle._label[idx])
            blob['voxels']  = [np.vstack(voxel_v[i]) for i in range(num_gpus)]
            blob['feature'] = [np.vstack(feature_v[i]) for i in range(num_gpus)]
            new_idx_v = [np.array(x) for x in new_idx_v]
            # if len(label_v): label_v = np.hstack(label_v)
            for key in io_handle._flags.DATA_KEYS:
                blob[key] = [np.vstack(minibatch) for minibatch in blob[key]]
            blob[io_handle._flags.DATA_KEYS[0]] = [np.concatenate([blob['voxels'][i], blob['feature'][i]], axis=1) for i in range(num_gpus)]
            io_handle._buffs[thread_id] = (new_idx_v, blob)
            io_handle._locks[thread_id] = True
    return

class io_larcv_sparse(io_base):

    def __init__(self, flags):
        super(io_larcv_sparse, self).__init__(flags=flags)
        self._fout    = None
        self._event_keys = []
        self._metas      = []
        # For circular buffer / thread function controls
        self._locks   = [False] * flags.NUM_THREADS
        self._buffs   = [None ] * flags.NUM_THREADS
        self._threads = [None ] * flags.NUM_THREADS
        self._start_idx = [-1 ] * flags.NUM_THREADS
        self._last_buffer_id = -1
        self.set_index_start(0)

    def initialize(self):
        pass

    def set_index_start(self,idx):
        self.stop_threads()
        for i in range(len(self._threads)):
            self._start_idx[i] = idx + i * self.batch_per_step()

    def start_threads(self):
        if self._threads[0] is not None:
            return
        for thread_id in range(len(self._threads)):
            print('Starting thread',thread_id)
            self._threads[thread_id] = threading.Thread(target = threadio_func, args=[self,thread_id])
            self._threads[thread_id].daemon = True
            self._threads[thread_id].start()

    def stop_threads(self):
        if self._threads[0] is None:
            return
        for i in range(len(self._threads)):
            while self._locks[buffer_id]:
                time.sleep(0.000001)
            self._buffs[i] = None
            self._start_idx[i] = -1

    def _next(self,buffer_id=-1,release=True):

        if buffer_id >= len(self._locks):
            sys.stderr.write('Invalid buffer id requested: {:d}\n'.format(buffer_id))
            raise ValueError
        if buffer_id < 0: buffer_id = self._last_buffer_id + 1
        if buffer_id >= len(self._locks):
            buffer_id = 0
        if self._threads[buffer_id] is None:
            sys.stderr.write('Read-thread does not exist (did you initialize?)\n')
            raise ValueError
        while not self._locks[buffer_id]:
            time.sleep(0.000001)
        res = self._buffs[buffer_id]
        if release:
            self._buffs[buffer_id] = None
            self._locks[buffer_id] = False
            self._last_buffer_id   = buffer_id

        return res

    def store_segment(self,idx_vv,data_vv,softmax_vv, **kwargs):
        for batch,idx_v in enumerate(idx_vv):
            start,end = (0,0)
            softmax_v = softmax_vv[batch]
            args_v = [kwargs[keyword][batch] for keyword in kwargs]
            for i,idx in enumerate(idx_v):
                voxels = self.blob()['voxels'][idx]
                end    = start + len(voxels)
                softmax = softmax_v[start:end,:]
                args_event = [arg_v[start:end, :] for arg_v in args_v]
                start = end
                self.store_one_segment(idx,softmax, **dict(zip(kwargs.keys(), args_event)))
            start = end

    def store_one_segment(self, idx, softmax, **kwargs):
        from larcv import larcv
        if self._fout is None:
            return
        idx=int(idx)
        if idx >= self.num_entries():
            raise ValueError
        keys = self._event_keys[idx]
        meta = self._metas[idx]

        data_key = self._flags.DATA_KEYS[0]

        larcv_data = self._fout.get_data('sparse3d',data_key)
        voxel   = self._blob['voxels'][idx]
        feature = self._blob['feature'][idx].reshape([-1])
        vs = larcv.as_tensor3d(voxel,feature,meta,0.)
        larcv_data.set(vs,meta)

        score = np.max(softmax,axis=1).reshape([-1])
        prediction = np.argmax(softmax,axis=1).astype(np.float32).reshape([-1])

        larcv_softmax = self._fout.get_data('sparse3d','softmax')
        vs = larcv.as_tensor3d(voxel,score,meta,-1.)
        larcv_softmax.set(vs,meta)

        larcv_prediction = self._fout.get_data('sparse3d','prediction')
        vs = larcv.as_tensor3d(voxel,prediction,meta,-1.)
        larcv_prediction.set(vs,meta)

        for keyword in kwargs:
            values = kwargs[keyword].reshape([-1]).astype(np.float32)
            larcv_arg = self._fout.get_data('sparse3d', keyword)
            vs = larcv.as_tensor3d(voxel, values, meta, -1.)
            larcv_arg.set(vs, meta)

        if len(self._flags.DATA_KEYS) > 1:
            label = self.blob()[self._flags.DATA_KEYS[1]][idx]
            label = label.astype(np.float32).reshape([-1])
            larcv_label = self._fout.get_data('sparse3d','label')
            vs = larcv.as_tensor3d(voxel,label,meta,-1.)
            larcv_label.set(vs,meta)
        self._fout.set_id(keys[0],keys[1],keys[2])
        self._fout.save_entry()

    def finalize(self):
        if self._fout:
            self._fout.finalize()
