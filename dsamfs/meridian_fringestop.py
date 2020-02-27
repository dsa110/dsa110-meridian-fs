"""
meridian_fringestopping.py
dana.simard@astro.caltech.edu, Feb 2020

This script reads correlated data from a psrdada
ringbuffer.  It fringestops on the meridian for each integrated
sample, before integrating the data and writing it to a hdf5 file.
"""

import numpy as np
from psrdada import Reader
from dsamfs.psrdada_utils import *
from dsamfs.fringestopping import *
from dsacalib.utils import *
from dsamfs.hdf5_utils import *
from antpos.utils import *
import h5py
import os
from datetime import datetime
import sys
import dsamfs

if len(sys.argv) > 1:
    param_file = sys.argv[1]
else:
    param_file = '{0}/meridian_fringestopping_parameters.py'.format(dsamfs.__path__[0])

import importlib.util
spec = importlib.util.spec_from_file_location('pm',param_file)
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)

#from meridian_fringestopping_parameters import *

# Get the visibility model
vis_model = load_visibility_model(pm.fs_table,pm.antenna_order,pm.nint,pm.nbls,pm.fobs)

if pm.test:
    sample_rate = 1/0.134217728
    header_size = 4096
    buffer_size = int(4*pm.nbls*pm.npol*pm.nchan*pm.samples_per_frame*2)
    data_rate = buffer_size*(sample_rate/pm.samples_per_frame)/1e6
    os.system('dada_db -a {0} -b {1} -k {2}'.format(
        header_size, buffer_size, pm.key_string))

print('Initializing reader: {0}'.format(pm.key_string))
reader = Reader(pm.key)

if pm.test:
    print('Writing data to psrdada buffer')
    os.system('dada_junkdb -r {0} -t 60 -k {2} {1}'.format(
        data_rate,'test_header.txt',pm.key_string))

# Get the start time and the sample time from the reader
#tstart, tsamp = read_header(reader)
#tstart       += pm.nint*tsamp/2
tstart = 58871.66878472222*ct.seconds_per_day
tsamp  = 0.134217728
tstart += pm.nint*tsamp/2
t0     = int(tstart)
tstart -= t0
sample_rate_out = 1/(tsamp*pm.nint)
nans = False

print('Opening output file {0}.hdf5'.format(pm.fname))
with h5py.File('{0}.hdf5'.format(pm.fname), 'w') as f:
    vis_ds, t_ds = initialize_hdf5_file(f,pm.fobs,pm.antenna_order,t0,
                                       pm.nbls,pm.nchan,pm.npol,pm.nant)
    
    idx_frame_out = 0
    while not nans:
        data_in = np.ones((pm.samples_per_frame_out*pm.nint,pm.nbls,pm.nchan,pm.npol),
                  dtype=np.complex64)*np.nan
        for i in range(data_in.shape[0]):
            try:
                assert reader.isConnected is True
                data_in[i,...] = read_buffer(reader,pm.nbls,pm.nchan,pm.npol)
            except:
                print('Last integration with only {0} timesamples'.format(i))
                nans = True
                break
        print(np.sum(data_in < 1.))
        data = fringestop_on_zenith(
            data_in,vis_model,pm.nint,nans)
        
        # Write out the data 
        t,tstart = update_time(tstart,
                               pm.samples_per_frame_out,sample_rate_out)
        f["vis"].resize((idx_frame_out+1)*pm.samples_per_frame_out,axis=0)
        f["time_seconds"].resize((idx_frame_out+1)*
                                 pm.samples_per_frame_out,axis=0)
        f["vis"][idx_frame_out*pm.samples_per_frame_out:,...]=data
        f["time_seconds"][idx_frame_out*pm.samples_per_frame_out:]=t

        idx_frame_out += 1
        print('Integration {0} done'.format(idx_frame_out))
        
    reader.disconnect() 
    if pm.test:
        os.system('dada_db -d -k {0}'.format(pm.key_string))

