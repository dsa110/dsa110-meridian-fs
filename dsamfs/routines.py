"""Routines for running meridian fringestopping of DSA-110 data.

Dana Simard, dana.simard@astro.caltech.edu, 2020
"""

import subprocess
import numpy as np
import astropy.units as u
from psrdada import Reader
import dsautils.dsa_syslog as dsl
import dsamfs.utils as pu
from dsamfs.io import dada_to_uvh5

logger = dsl.DsaSyslogger()
logger.subsystem("software")
logger.app("dsamfs")

def run_fringestopping(param_file=None, header_file=None, output_dir=None):
    """Read in data, fringestop on zenith, and write to hdf5 file.
    Parameters
    ----------
    param_file : str
        The full path to the json parameter file. Defaults to the file
        meridian_fringestopping_parameters.py in the package directory.
    """
    # Read in parameter file
    test, key_string, nant, nchan, npol, fobs, samples_per_frame, \
        samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays = \
        pu.parse_params(param_file)
    nbls = (nant*(nant+1))//2
    key = int('0x{0}'.format(key_string), 16)

    fs_table = 'fringestopping_table_dec{0:.1f}deg_{1}ant.npz'.format((pt_dec*u.rad).to_value(u.deg), len(antenna_order))
    if output_dir is not None:
        fs_table = '{0}/{1}'.format(output_dir, fs_table)
    bname, blen, uvw = pu.baseline_uvw(antenna_order, pt_dec, casa_order=False)

    logger.info("Started fringestopping of dada buffer {0} with {1} "
                "integrations and {2} baselines.")

    # Get the visibility model
    vis_model = pu.load_visibility_model(
        fs_table, blen, nint, fobs, pt_dec, tsamp, antenna_order,
        outrigger_delays, bname
    )
    if not fringestop:
        vis_model = np.ones(vis_model.shape, vis_model.dtype)

    if test:
        sample_rate = 1/0.134217728
        header_size = 4096
        buffer_size = int(4*nbls*npol*nchan*samples_per_frame*2)
        data_rate = buffer_size*(sample_rate/samples_per_frame)/1e6
        p_create = subprocess.Popen(
            ["dada_db", "-a", str(header_size), "-b", str(buffer_size), "-k",
             key_string], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        outs, errs = p_create.communicate(timeout=15)
        if p_create.returncode != 0:
            print(errs.decode("utf-8"))
            logger.info(errs.decode("utf-8"))
            raise RuntimeError('Dada buffer could not be created.')
        print(outs.decode("utf-8"))

    print('Initializing reader: {0}'.format(key_string))
    reader = Reader(key)

    if test:
        p_write = subprocess.Popen(
            ["dada_junkdb", "-r", str(data_rate), "-t", "60", "-k", key_string,
             header_file], stdout=subprocess.PIPE,
             stderr=subprocess.PIPE)

    # Get the start time and the sample time from the reader
    sample_rate_out = 1/(tsamp*nint)

    # Read in psrdada buffer, fringestop, and write to uvh5
    dada_to_uvh5(
        reader, output_dir, nbls, nchan, npol, nint, nfreq_int,
        samples_per_frame_out, sample_rate_out, pt_dec, antenna_order,
        fs_table, tsamp, bname, uvw, fobs,
        vis_model, test, filelength_minutes
    )

    if test:
        outs, errs = p_write.communicate(timeout=15)
        if p_write.returncode != 0:
            logger.info(errs.decode("utf-8"))
            print(errs.decode("utf-8"))
            raise RuntimeError('Error in writing to dada buffer.')
        print(outs.decode("utf-8"))
        print(errs.decode("utf-8"))
        p_kill = subprocess.Popen(
            ["dada_db", "-d", "-k", key_string], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        outs, errs = p_kill.communicate(timeout=15)
        if p_kill.returncode != 0:
            logger.info(errs.decode("utf-8"))
            print(errs.decode("utf-8"))
        else:
            print(outs.decode("utf-8"))

    logger.info("Disconnected from psrdada buffer {0}".format(key_string))
