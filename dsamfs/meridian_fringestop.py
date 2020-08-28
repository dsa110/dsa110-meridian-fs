"""
meridian_fringestopping.py
dana.simard@astro.caltech.edu, Feb 2020

This script reads correlated data from a psrdada
ringbuffer.  It fringestops on the meridian for each integrated
sample, before integrating the data and writing it to a hdf5 file.
"""

import sys
import os
import json
import numpy as np
import h5py
from psrdada import Reader
import dsacalib.constants as ct
import dsamfs.psrdada_utils as pu
from dsamfs.fringestopping import fringestop_on_zenith
from dsacalib.hdf5_io import initialize_hdf5_file
import dsamfs
import dsautils.dsa_syslog as dsl

logger = dsl.DsaSyslogger()
logger.subsystem("software")
logger.app("dsamfs")

def parse_param_file(param_file):
    """Parses parameter file.

    Params
    ------
    param_file : str
        The full path to the json parameter file.
    """
    fhand = open(param_file)
    params = json.load(fhand)
    fhand.close()
    test = params['test']
    key_string = params['key_string']
    nant = params['nant']
    nchan = params['nchan']
    npol = params['npol']
    fout = params['hdf5_fname']
    samples_per_frame = params['samples_per_frame']
    samples_per_frame_out = params['samples_per_frame_out']
    nint = params['nint']
    fs_table = params['fs_table']
    antenna_order = params['antenna_order']
    dfreq = params['bw_GHz']/nchan
    fobs = params['f0_GHz']+dfreq/2+np.arange(nchan)*dfreq
    if not params['chan_ascending']:
        fobs = fobs[::-1]

    assert (samples_per_frame_out*nint)%samples_per_frame == 0, \
        "Each frame out must contain an integer number of frames in."

    return test, key_string, nant, nchan, npol, fobs, fout, \
        samples_per_frame, samples_per_frame_out, nint, fs_table, antenna_order

def main(param_file):
    """Read in data, fringestop on zenith, and write to hdf5 file.
    Parameters
    ----------
    param_file : str
        The full path to the json parameter file. Defaults to the file
        meridian_fringestopping_parameters.py in the package directory.
    """
    if param_file is None:
        param_file = '{0}/meridian_fringestopping_parameters.json'.format(
            dsamfs.__path__[0])
    test, key_string, nant, nchan, npol, fobs, fout, samples_per_frame, \
        samples_per_frame_out, nint, fs_table, antenna_order = \
        parse_param_file(param_file)
    nbls = (nant*(nant+1))//2
    key = int('0x{0}'.format(key_string), 16)
    # Get the visibility model
    vis_model = pu.load_visibility_model(fs_table, antenna_order, nint, nbls,
                                         fobs)
    logger.info("Started fringestopping of dada buffer {0} with {1} "
                "integrations and {2} baselines. Fringestopped data written "
                "to {2}.hdf5".format(key_string, nint, nbls, fout))
    
    if test:
        sample_rate = 1/0.134217728
        header_size = 4096
        buffer_size = int(4*nbls*npol*nchan*samples_per_frame*2)
        data_rate = buffer_size*(sample_rate/samples_per_frame)/1e6
        print('dada_db -a {0} -b {1} -k {2}'.format(
            header_size, buffer_size, key_string))
        input("Press Enter to continue...")

    print('Initializing reader: {0}'.format(key_string))
    reader = Reader(key)

    if test:
        #print('Writing data to psrdada buffer.')
        print('dada_junkdb -r {0} -t 60 -k {2} {1}'.format(
            data_rate, 'test_header.txt', key_string))
        input("Press Enter to continue...")

    # Get the start time and the sample time from the reader
    # Are these currently in the reader??
    # tstart, tsamp = pu.read_header(reader)
     #tstart += nint*tsamp/2
    tstart = 58871.66878472222*ct.SECONDS_PER_DAY
    tsamp = 0.134217728
    tstart += nint*tsamp/2
    t0 = int(tstart)
    tstart -= t0
    sample_rate_out = 1/(tsamp*nint)
    nans = False

    print('Opening output file {0}.hdf5'.format(fout))
    with h5py.File('{0}.hdf5'.format(fout), 'w') as fhdf5:
        _vis_ds, _t_ds = initialize_hdf5_file(fhdf5, fobs, antenna_order, t0,
                                              nbls, nchan, npol, nant)

        idx_frame_out = 0
        while not nans:
            data_in = np.ones((samples_per_frame_out*nint, nbls, nchan, npol),
                              dtype=np.complex64)*np.nan
            for i in range(data_in.shape[0]):
                try:
                    assert reader.isConnected
                    data_in[i, ...] = pu.read_buffer(reader, nbls, nchan, npol)
                except:
                    print('Last integration has {0} timesamples'.format(i))
                    nans = True
                    break

            data = fringestop_on_zenith(data_in, vis_model, nans)

            # Write out the data
            t, tstart = pu.update_time(tstart, samples_per_frame_out,
                                       sample_rate_out)
            fhdf5["vis"].resize((idx_frame_out+1)*samples_per_frame_out,
                                axis=0)
            fhdf5["time_seconds"].resize((idx_frame_out+1)*
                                         samples_per_frame_out, axis=0)
            fhdf5["vis"][idx_frame_out*samples_per_frame_out:, ...] = data
            fhdf5["time_seconds"][idx_frame_out*samples_per_frame_out:] = t

            idx_frame_out += 1
            print('Integration {0} done'.format(idx_frame_out))

        reader.disconnect()
        if test:
            print('dada_db -d -k {0}'.format(key_string))
            input("Press Enter to continue...")
        
        logger.info("Disconnected from psrdada buffer {0} and closed file "
                    "{1}.hdf5".format(key_string, fout))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        PARAM_FILE = sys.argv[1]
    else:
        PARAM_FILE = None
    main(PARAM_FILE)
