from ROOT import TFile
import numpy as np
import h5py
import sys
import os

'''Utilities for converting ArgonCube style data
into 'Kazu'-style data
'''


light_baryons = [2212, 2112, 2224, 2214, 2114, 1114]
strange_baryons = [3112, 3122, 3222, 3212, 3113, 3224, 3214, 
                   3114, 3322, 3312, 3324, 3314, 3334]
charmed_baryons = [4122, 4222, 4212, 4112, 4224, 4214, 
                   4114, 4232, 4132, 4322, 4312, 4324, 
                   4314, 4332, 4334, 4412, 4422, 4414, 
                   4424, 4434, 4444]
baryons = light_baryons + strange_baryons + charmed_baryons

kazu_codes = {
    'michel': 0,
    'delta': 1,
    'shower': 2,
    'hip': 3,
    'mip': 4
}

def pdg_to_kazu_coding(pdg):
    if pdg == 11: # simplification, fix later
        return kazu_codes['shower']
    elif -pdg == 11:
        return kazu_codes['shower']
    elif pdg > 1e9 or abs(pdg) in baryons:
        return kazu_codes['hip']
    elif pdg in [-13, 13, 111, 211, 321, -211, -321]:
        return kazu_codes['mip']
    else:
        raise Exception('Unidentified PDG %d' % pdg)

def load_argon_tree(filename):
    arcube = TFile(filename)
    return arcube.Get('argon')

def convert_event_data(event):
    x, y, z = list(event.xq), list(event.yq), list(event.zq)
    pid, energies = list(event.pidq), list(event.dq)
    labels = [pdg_to_kazu_coding(p) for p in pid]
    
    return x, y, z, energies, labels

def get_entries_per_event(argon, start, end):
    num_entries_per_event = []
    for i in range(start, end):
        argon.GetEntry(i)
        num_entries_per_event.append(len(argon.xq))
    return num_entries_per_event

def raw_arcube_file_loader(argon, batch_size):
    for start_index in range(0, 1000 // batch_size, batch_size):
        num_entries_per_event = get_entries_per_event(argon, start_index, start_index+batch_size)
        total_entries = sum(num_entries_per_event)
        coordinates = np.empty((total_entries, 4))
        features = np.empty((total_entries, 1))
        labels = np.empty((total_entries, 1))
        c_ind = 0
        '''
        i = 0
        for ev in argon:
            if i in list(range(start_index, start_index+batch_size)):
                end = c_ind+num_entries_per_event[i-start_index]
                x, y, z = list(ev.xq), list(ev.yq), list(ev.zq)
                pid, energies = list(ev.pidq), list(ev.dq)
                _labels = [pdg_to_kazu_coding(p) for p in pid]
                coordinates[c_ind: end, 0] = x
                coordinates[c_ind: end, 1] = y
                coordinates[c_ind: end, 2] = z
                coordinates[c_ind: end, 3] = i-start_index

                features[c_ind: end] = np.array(energies).reshape((len(energies), 1))
                labels[c_ind: end] = np.array(_labels).reshape((len(_labels), 1))
                c_ind += len(x)
            i += 1
        '''
        for i in range(start_index, start_index+batch_size):
            argon.GetEntry(i)
            end = c_ind+num_entries_per_event[i-start_index]
            x, y, z = list(argon.xq), list(argon.yq), list(argon.zq)
            pid, energies = list(argon.pidq), list(argon.dq)
            _labels = [pdg_to_kazu_coding(p) for p in pid]
            coordinates[c_ind: end, 0] = x
            coordinates[c_ind: end, 1] = y
            coordinates[c_ind: end, 2] = z
            coordinates[c_ind: end, 3] = i-start_index
            features[c_ind: end] = np.array(energies).reshape((len(energies), 1))
            labels[c_ind: end] = np.array(_labels).reshape((len(_labels), 1))
            c_ind += len(x)

        yield -1, coordinates, features, labels

def inside_region(voxel, dimension):
    '''Returns True if voxel in cubic volume
    of side-length dimension centered at (0,0,0)
    '''
    for coord in voxel[0:3]:
        if abs(coord) >= (dimension-1)/2:
            return False
    return True

def voxelize(coordinates, features, labels, dimension=192):
    '''Convert discrete data from ArCube into voxelized data
    with resolution 0.3cm (so that a voxel 0.3cm on side)
    '''
    uncentered_voxels = (coordinates / 0.3).astype('int')
    rows_in_region, features_in_region, labels_in_region = [], [], []
    for row in range(uncentered_voxels.shape[0]):
        if inside_region(uncentered_voxels[row], dimension):
            rows_in_region.append(uncentered_voxels[row])
            features_in_region.append(features[row])
            labels_in_region.append(labels[row])
    legal_voxels = np.array(rows_in_region)
    legal_features = np.array(features_in_region)
    legal_labels = np.array(labels_in_region)
    return legal_voxels + np.full(legal_voxels.shape, (dimension-1)/2), legal_features, legal_labels
    
def voxelized_arcube_file_loader(argon, batch_size, dimension):
    for _, c, f, l in raw_arcube_file_loader(argon, batch_size):
        voxels, features, labels = voxelize(c, f, l, dimension)
        if c.shape[0] > 0 and voxels.shape[0] == 0:
            # entire event lies outside region
            # (but why/how is this possible?)
            continue
        yield dimension, voxels, features, labels

def ArCube_to_HDF5(filename, dimension, noisy=False):
    arcube = TFile(filename)
    argon = arcube.Get('argon')
    n_events = len([ev for ev in argon])
    dt = h5py.special_dtype(vlen=np.dtype('float32'))
    if filename[-5:] == '.root':
        filename = filename[:-5]
    if noisy:
        print('Converting %s.root to HDF5...' % filename)
    with h5py.File(filename+'.hdf5', 'w') as f:
        _dimension = f.create_dataset('dimension', (1,), dtype=np.dtype('int32'))
        _dimension = dimension
        voxels_x = f.create_dataset('voxels_x', (n_events,), dtype=dt)
        voxels_y = f.create_dataset('voxels_y', (n_events,), dtype=dt)
        voxels_z = f.create_dataset('voxels_z', (n_events,), dtype=dt)
        energies = f.create_dataset('energies', (n_events,), dtype=dt)
        labels = f.create_dataset('labels', (n_events,), dtype=dt)
        i = 0
        for d, v, f, l in voxelized_arcube_file_loader(argon, 1, dimension):
            if i % 100 == 0 and noisy:
                print('\t%d percent complete.' % (100.0 * (i+1)/n_events))
            voxels_x[i] = v[:, 0]
            voxels_y[i] = v[:, 1]
            voxels_z[i] = v[:, 2]
            energies[i] = f.flatten()
            labels[i] = l.flatten()
            i += 1
        print("\t100 percent complete.")
        
if __name__ == '__main__':
    ok = False
    while not ok:
        src_dir = raw_input("\nWhat directory would you like to convert (ArCube -> HDF5)? ")
        if src_dir[-1] != '/':
            src_dir += '/'
        print('%s contains...' % src_dir)
        for f in os.listdir(src_dir):
            print('\t'+f)
        ok = raw_input("Is this what you want? ").lower() in ['yes', 'y', 'yep', '']
    dimension = int(raw_input("\nWhat dimension? "))
    for f in os.listdir(src_dir):
        if not os.path.isfile(src_dir + f):
            continue
        elif f[-5:] != '.root':
            continue
        elif f[-5:] == '.hdf5':
            continue
        elif f[:-5] + '.hdf5' in os.listdir(src_dir):
            continue

        ArCube_to_HDF5(src_dir + f, dimension, True)
