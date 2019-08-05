import numpy as np
import h5py

'''This file contains HDF5_to_standard_np and HDF5_to_nz_np which
load data from HDF5 into numpy, but in different ways.

HDF5_to_standard_np loads HDF5 data into what I refer to as the
'standard' representation for a tensor. It returns a full-size
tensor with an entry in every place of the tensor, even if the
entry is 0.

HDF5_to_nz_np loads HDF5 data into what I refer to as the 
'nonzero' representation for a tensor. Instead of return a full-size
tensor, the function returns the dimensions of the tensor and
a list of only the nonzero entries of the tensor and the corresponding
values of those entries.
'''

def HDF5_data_dimension(filename):
    '''Returns size of data stored in HDF5 file.
    '''
    with h5py.File(filename, 'r') as f:
        return f['dimension'][0]

def load_and_convert_HDF5_to_standard_np(filename, start_index, batch_size):
    '''Load HDF5 data into a numpy tensor in standard representation.

    Keyword arguments:
    filename -- HDF5 file to load from
    start_index -- first instance number to load
    batch_size -- number of instances to convert to sparse
                  numpy arrays
    '''
    with h5py.File(filename, 'r') as f:
        dim = f['dimension'][0]
        voxels_x = f['voxels_x']
        voxels_y = f['voxels_y']
        voxels_z = f['voxels_z']

        energies = f['energies']
        _labels = f['labels']

        data = np.empty((batch_size, dim, dim, dim))
        labels = np.empty((batch_size, dim, dim, dim))
        for i in range(start_index, start_index+batch_size):
            x, y, z = [j[i].astype(np.int) for j in [voxels_x, voxels_y, voxels_z]]
            data[i-start_index, x, y, z] = energies[i]
            labels[i-start_index, x, y, z] = _labels[i]
    return data, labels
        
def load_and_convert_HDF5_to_nz_np(filename, start_index, batch_size):
    '''Load HDF5 data into a numpy tensor in 'nonzero' representation.

    Keyword arguments:
    filename -- HDF5 file to load from
    start_index -- first instance number to load
    batch_size -- number of instances to convert to sparse
                  numpy arrays
    '''
    with h5py.File(filename, 'r') as f:
        dim = f['dimension'][0]
        voxels_x = f['voxels_x']
        voxels_y = f['voxels_y']
        voxels_z = f['voxels_z']

        energies = f['energies']
        _labels = f['labels']

        num_entries_per_event = [len(voxels_x[i]) for i in range(start_index, start_index+batch_size)]
        total_entries = sum(num_entries_per_event)
        coordinates = np.empty((total_entries, 4))
        features = np.empty((total_entries, 1))
        labels = np.empty((total_entries, 1))

        c_ind = 0

        for i in range(start_index, start_index+batch_size):
            end = c_ind+num_entries_per_event[i-start_index]
            coordinates[c_ind: end, 0] = voxels_x[i]
            coordinates[c_ind: end, 1] = voxels_y[i]
            coordinates[c_ind: end, 2] = voxels_z[i]
            coordinates[c_ind: end, 3] = i-start_index
            
            features[c_ind: end] = energies[i].reshape((energies[i].shape[0], 1))
            labels[c_ind: end] = _labels[i].reshape((_labels[i].shape[0], 1))
            
            c_ind += len(voxels_x[i])

        return dim, coordinates, features, labels

def linear_train_loader(filename, batch_size):
    '''Generator that yields numpy data converted from HDF5 batchwise.
    '''
    with h5py.File(filename, 'r') as f:
        dim = f['dimension'][0]
        voxels_x = f['voxels_x']
        voxels_y = f['voxels_y']
        voxels_z = f['voxels_z']
        energies = f['energies']
        _labels = f['labels']
        
        for start_index in range(0, 10000//batch_size, batch_size):
            num_entries_per_event = [len(voxels_x[i]) for i in range(start_index, start_index+batch_size)]
            total_entries = sum(num_entries_per_event)
            coordinates = np.empty((total_entries, 4))
            features = np.empty((total_entries, 1))
            labels = np.empty((total_entries, 1))

            c_ind = 0

            for i in range(start_index, start_index+batch_size):
                end = c_ind+num_entries_per_event[i-start_index]
                coordinates[c_ind: end, 0] = voxels_x[i]
                coordinates[c_ind: end, 1] = voxels_y[i]
                coordinates[c_ind: end, 2] = voxels_z[i]
                coordinates[c_ind: end, 3] = i-start_index
                
                features[c_ind: end] = energies[i].reshape((energies[i].shape[0], 1))
                labels[c_ind: end] = _labels[i].reshape((_labels[i].shape[0], 1))

                c_ind += len(voxels_x[i])

            yield dim, coordinates, features, labels, num_entries_per_event
